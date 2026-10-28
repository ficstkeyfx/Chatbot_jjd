"""Account-owned bot workers; tokens stay inside server storage and transport."""
import asyncio
import re

from storage import StorageError
from zalo_bridge import ZaloAPI, ZaloBridge


class AccountBots:
    def __init__(self, store, shared):
        self.store, self.shared = store, shared
        self.shared_configured = shared.enabled
        self.bots = {}
        self.lock = asyncio.Lock()
        self.task = None

    def bridge_for(self, account_id):
        return self.bots.get(account_id, self.shared)

    async def start(self):
        await self.reconcile()
        self.task = asyncio.create_task(self.watch())

    async def reconcile(self):
        if self.store is None:
            return
        async with self.lock:
            configs = {r['account_id']: r for r in self.store.list_zalo_bots()}
            for account_id, bridge in list(self.bots.items()):
                row = configs.get(account_id)
                if not row or row['token'] != bridge.token:
                    await bridge.stop()
                    del self.bots[account_id]
            shared_id = self.shared.token.split(':', 1)[0]
            shared_active = self.shared_configured and not any(r['bot_id'] == shared_id for r in configs.values())
            if not shared_active and self.shared.enabled:
                await self.shared.stop()
                self.shared.enabled = False
            elif shared_active and not self.shared.tasks:
                self.shared.enabled = True
                self.shared.api = None
                await self.shared.start()
            for account_id, row in configs.items():
                if account_id not in self.bots:
                    bridge = ZaloBridge(self.store, owner_id=account_id, bot_id=row['bot_id'], token=row['token'])
                    self.bots[account_id] = bridge
                    await bridge.start()

    async def watch(self):
        while True:
            await asyncio.sleep(5)
            await self.reconcile()

    async def configure(self, account_id, token):
        if not isinstance(token, str) or not re.fullmatch(r'\d+:[A-Za-z0-9_-]{10,480}', token.strip()):
            raise ValueError('Token bot không đúng định dạng.')
        token = token.strip()
        api = ZaloAPI(token)
        try:
            info = await api.call('getMe')
            if not isinstance(info, dict) or not isinstance(info.get('id'), str) or not info['id'].isdigit():
                raise ValueError('Không xác minh được bot. Hãy kiểm tra token.')
            try:
                webhook = await api.call('getWebhookInfo')
            except RuntimeError as exc:
                if '(404)' not in str(exc):
                    raise
                webhook = None
            if isinstance(webhook, dict) and webhook.get('url'):
                raise ValueError('Bot đang dùng webhook. Hãy gỡ webhook trước khi kết nối bot tại đây.')
            self.store.save_zalo_bot(account_id, info['id'], str(info.get('display_name') or info.get('account_name') or 'Bot Zalo')[:200], token)
        finally:
            await api.close()
        await self.reconcile()

    async def remove(self, account_id):
        async with self.lock:
            bridge = self.bots.pop(account_id, None)
            if bridge:
                await bridge.stop()
            self.store.remove_zalo_bot(account_id)
        await self.reconcile()

    async def notify(self, sid, event):
        if sid.startswith('acct-'):
            bridge = self.bots.get(int(sid[5:]))
            if bridge:
                await bridge.notify(sid, event)

    async def stop(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        for bridge in self.bots.values():
            await bridge.stop()
        self.bots.clear()
