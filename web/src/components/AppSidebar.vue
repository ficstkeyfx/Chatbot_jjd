<script setup>
import { FileText, Heart, LogOut, MessageSquarePlus, Phone, Settings, Smile, UserRound, Wind, X } from '@lucide/vue'
import PomodoroCard from './PomodoroCard.vue'
import EyeCard from './EyeCard.vue'
import { auth, logout } from '../composables/useAuth'
import { newConversation, sendText } from '../composables/useChat'
import { openDialog, requestPick, ui } from '../composables/useUi'

function breathing() {
  ui.drawer = false
  sendText('Hướng dẫn tôi bài tập thở chậm để bình tĩnh lại')
}
</script>

<template>
  <aside class="sidebar" :class="{ open: ui.drawer }" aria-label="Công cụ">
    <div class="brand">
      <span class="logo"><Heart :size="22" fill="currentColor" /></span>
      <div>
        <strong>Trợ lý Sức khỏe</strong>
        <small>Giảm kiệt quệ kỹ thuật số</small>
      </div>
      <button class="icon-btn close" aria-label="Đóng thanh bên" @click="ui.drawer = false">
        <X :size="20" />
      </button>
    </div>

    <button class="btn btn-primary new" @click="newConversation(); ui.drawer = false">
      <MessageSquarePlus :size="18" /> Cuộc trò chuyện mới
    </button>

    <div class="scroll">
      <h4>Công cụ nhanh</h4>
      <div class="tools">
        <button class="tool" @click="openDialog('summary')">
          <span class="ico" style="--c: var(--primary-soft); --f: var(--primary-strong)"><FileText :size="18" /></span>
          <span><strong>Tóm tắt bài học</strong><small>Rút gọn còn 3 ý chính</small></span>
        </button>
        <button class="tool" @click="requestPick(true)">
          <span class="ico" style="--c: var(--accent-soft); --f: var(--accent)"><Smile :size="18" /></span>
          <span><strong>Biểu cảm &amp; tâm lý</strong><small>Gửi ảnh để nhận gợi ý hỗ trợ</small></span>
        </button>
        <button class="tool" @click="breathing">
          <span class="ico" style="--c: var(--info-soft); --f: #3a86d1"><Wind :size="18" /></span>
          <span><strong>Thở chậm 4-6</strong><small>Bình tĩnh lại trong 1 phút</small></span>
        </button>
      </div>

      <h4>Hẹn giờ</h4>
      <div class="timers">
        <PomodoroCard />
        <EyeCard />
      </div>
    </div>

    <div v-if="auth.required && auth.user" class="user">
      <span class="avatar"><UserRound :size="16" /></span>
      <strong :title="auth.user.username">{{ auth.user.username }}</strong>
      <button class="icon-btn" aria-label="Đăng xuất" title="Đăng xuất" @click="logout">
        <LogOut :size="17" />
      </button>
    </div>

    <footer>
      <a class="sos" href="tel:115"><Phone :size="16" /> Cấp cứu 115</a>
      <button class="btn btn-ghost" @click="openDialog('settings')"><Settings :size="17" /> Cài đặt</button>
    </footer>
  </aside>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-w);
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px 16px 14px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  height: 100%;
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand div {
  display: grid;
  line-height: 1.25;
  flex: 1;
}
.brand small {
  color: var(--text-soft);
  font-size: 12.5px;
}
.logo {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--primary), #3fd0b0);
  color: #fff;
  box-shadow: 0 6px 16px rgba(15, 157, 131, 0.35);
}
.close {
  display: none;
}
.new {
  width: 100%;
  padding: 11px 16px;
}
.scroll {
  flex: 1;
  overflow-y: auto;
  margin: 0 -6px;
  padding: 0 6px;
}
h4 {
  margin: 14px 0 8px;
  font-size: 11.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
h4:first-child {
  margin-top: 2px;
}
.tools,
.timers {
  display: grid;
  gap: 8px;
}
.tool {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  text-align: left;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
  transition: border-color 0.15s, transform 0.12s, box-shadow 0.15s;
}
.tool:hover {
  border-color: var(--primary);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}
.tool .ico {
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 11px;
  background: var(--c);
  color: var(--f);
  flex: none;
}
.tool span:last-child {
  display: grid;
  line-height: 1.3;
}
.tool small {
  color: var(--text-soft);
  font-size: 12.5px;
}
.user {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 8px 8px 12px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface-2);
}
.user strong {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
}
.user .avatar {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--border);
}
.sos {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 14px;
  border-radius: 12px;
  background: var(--danger-soft);
  color: var(--danger);
  font-weight: 700;
  font-size: 14px;
  text-decoration: none;
}
.sos:hover {
  filter: brightness(0.96);
}

@media (max-width: 900px) {
  .sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 40;
    width: min(340px, 88vw);
    transform: translateX(-102%);
    transition: transform 0.25s ease;
    box-shadow: var(--shadow-lg);
  }
  .sidebar.open {
    transform: none;
  }
  .close {
    display: grid;
  }
}
</style>
