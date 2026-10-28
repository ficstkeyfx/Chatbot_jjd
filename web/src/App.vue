<script setup>
import { computed, onBeforeUnmount, onMounted, watch } from 'vue'
import { Menu, Moon, Sun } from '@lucide/vue'
import AppSidebar from './components/AppSidebar.vue'
import AuthView from './components/AuthView.vue'
import Composer from './components/Composer.vue'
import MessageList from './components/MessageList.vue'
import SettingsDialog from './components/SettingsDialog.vue'
import SummaryDialog from './components/SummaryDialog.vue'
import ToastHost from './components/ToastHost.vue'
import { auth, isSignedIn, loadAuth } from './composables/useAuth'
import { addSystemMessage, loadHistory, resetChat, syncRemoteHistory } from './composables/useChat'
import { loadSettings, providerById, resetSettings, settings } from './composables/useSettings'
import {
  connectEvents, disconnectEvents, onReminder, onSync, refreshTimers, resetTimers,
} from './composables/useTimers'
import { toast } from './composables/useToast'
import { openDialog, toggleTheme, ui } from './composables/useUi'

const signedIn = computed(() => isSignedIn())
const needsLogin = computed(() => auth.ready && auth.required && !auth.user)

const status = computed(() => {
  if (!settings.loaded) return { text: 'Đang kết nối…', ok: false }
  const label = providerById(settings.profile.llm.provider)?.label.split(' (')[0] || settings.profile.llm.provider
  return { text: `${label} · ${settings.effective.llm || 'chưa chọn model'}`, ok: true }
})

async function startSession() {
  try {
    await loadSettings()
  } catch (error) {
    if (error.status !== 401) toast(error.message, { type: 'error', duration: 8000 })
    return
  }
  await Promise.all([loadHistory().catch(() => {}), refreshTimers()])
  connectEvents()
}

// Đăng xuất hoặc hết phiên: dọn sạch để tài khoản sau không thấy dữ liệu của tài khoản trước.
function endSession() {
  disconnectEvents()
  resetChat()
  resetSettings()
  resetTimers()
  ui.dialog = null
  ui.drawer = false
}

watch(signedIn, (now) => (now ? startSession() : endSession()))

onMounted(async () => {
  onReminder((event) => addSystemMessage(event.text))
  onSync(async () => {
    try {
      await Promise.all([syncRemoteHistory(), loadSettings()])
    } catch { /* SSE reconnect will retry synchronization. */ }
  })
  try {
    await loadAuth()
  } catch (error) {
    toast(error.message, { type: 'error', duration: 8000 })
  }
})
onBeforeUnmount(disconnectEvents)
</script>

<template>
  <div v-if="!auth.ready" class="splash" aria-busy="true">Đang tải…</div>
  <template v-else-if="needsLogin">
    <AuthView />
    <ToastHost />
  </template>
  <div v-else class="app">
    <AppSidebar />
    <div v-if="ui.drawer" class="scrim" @click="ui.drawer = false" />

    <main class="main">
      <header class="topbar">
        <button class="icon-btn menu" aria-label="Mở thanh công cụ" @click="ui.drawer = true">
          <Menu :size="22" />
        </button>
        <div class="title">
          <h1>Trợ lý Sức khỏe</h1>
          <button class="status" :class="{ ok: status.ok }" title="Mở cài đặt" @click="openDialog('settings')">
            <i /> {{ status.text }}
          </button>
        </div>
        <button
          class="icon-btn"
          :aria-label="ui.theme === 'dark' ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'"
          @click="toggleTheme"
        >
          <Sun v-if="ui.theme === 'dark'" :size="20" />
          <Moon v-else :size="20" />
        </button>
      </header>

      <MessageList />
      <Composer />
    </main>

    <SettingsDialog v-if="ui.dialog === 'settings'" />
    <SummaryDialog v-if="ui.dialog === 'summary'" />
    <ToastHost />
  </div>
</template>

<style scoped>
.app {
  display: flex;
  height: 100%;
  background:
    radial-gradient(900px 400px at 85% -10%, color-mix(in srgb, var(--primary) 12%, transparent), transparent),
    var(--bg);
}
.main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  height: 100%;
}
.topbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--bg) 80%, transparent);
  backdrop-filter: blur(8px);
}
.title {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px 12px;
}
h1 {
  margin: 0;
  font-size: 17px;
}
.status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  max-width: 100%;
  padding: 3px 11px;
  border-radius: 99px;
  background: var(--surface);
  border: 1px solid var(--border);
  font-size: 12.5px;
  color: var(--text-soft);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.status:hover {
  border-color: var(--primary);
}
.status i {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--muted);
}
.status.ok i {
  background: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-soft);
}
.menu {
  display: none;
}
.splash {
  height: 100%;
  display: grid;
  place-items: center;
  color: var(--text-soft);
}
.scrim {
  position: fixed;
  inset: 0;
  z-index: 30;
  background: rgba(10, 25, 22, 0.45);
  animation: fade 0.2s;
}
@keyframes fade {
  from {
    opacity: 0;
  }
}
@media (max-width: 900px) {
  .menu {
    display: grid;
  }
  .topbar {
    padding: 10px 12px;
  }
}
</style>
