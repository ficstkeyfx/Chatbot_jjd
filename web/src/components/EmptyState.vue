<script setup>
import { Eye, Heart, Moon, Wind } from '@lucide/vue'
import { sendText } from '../composables/useChat'

const suggestions = [
  { icon: Eye, title: 'Mỏi mắt', text: 'Tôi học máy tính 4 tiếng rồi, mắt rất mỏi.' },
  { icon: Wind, title: 'Giảm căng thẳng', text: 'Tôi đang thấy căng thẳng, mình nên làm gì cho bình tĩnh lại?' },
  { icon: Moon, title: 'Khó ngủ', text: 'Tôi khó ngủ thì làm sao?' },
  { icon: Heart, title: 'Giảm lệ thuộc điện thoại', text: 'Làm sao để bớt lướt điện thoại vào buổi tối?' },
]
</script>

<template>
  <div class="empty">
    <span class="hero"><Heart :size="34" fill="currentColor" /></span>
    <h1>Xin chào, mình có thể giúp gì cho bạn?</h1>
    <p>
      Hỏi về sức khỏe, giấc ngủ, mỏi mắt hay áp lực học tập. Bạn cũng có thể gửi ảnh, ghi âm giọng nói
      hoặc dán bài học để mình tóm tắt.
    </p>
    <div class="grid">
      <button v-for="item in suggestions" :key="item.title" class="card" @click="sendText(item.text)">
        <span class="ico"><component :is="item.icon" :size="18" /></span>
        <span>
          <strong>{{ item.title }}</strong>
          <small>{{ item.text }}</small>
        </span>
      </button>
    </div>
    <p class="note">
      Mình chỉ hỗ trợ ban đầu, không chẩn đoán và không thay thế bác sĩ. Khi có dấu hiệu nguy hiểm, hãy gọi
      <a href="tel:115">115</a>.
    </p>
  </div>
</template>

<style scoped>
.empty {
  margin: auto;
  width: min(720px, 100%);
  display: grid;
  justify-items: center;
  text-align: center;
  gap: 8px;
  padding: 24px 0;
}
.hero {
  display: grid;
  place-items: center;
  width: 72px;
  height: 72px;
  border-radius: 24px;
  color: #fff;
  background: linear-gradient(135deg, var(--primary), #3fd0b0);
  box-shadow: 0 14px 32px rgba(15, 157, 131, 0.35);
  margin-bottom: 8px;
}
h1 {
  margin: 0;
  font-size: clamp(22px, 3.4vw, 30px);
  line-height: 1.25;
}
p {
  margin: 0;
  max-width: 560px;
  color: var(--text-soft);
}
.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  width: 100%;
  margin: 18px 0 8px;
}
.card {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px;
  text-align: left;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  transition: transform 0.12s, border-color 0.15s, box-shadow 0.15s;
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--primary);
  box-shadow: var(--shadow);
}
.ico {
  flex: none;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 11px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.card span:last-child {
  display: grid;
  gap: 2px;
  min-width: 0;
}
.card small {
  color: var(--text-soft);
  font-size: 13px;
  line-height: 1.4;
}
.note {
  font-size: 12.5px;
  color: var(--muted);
}
.note a {
  color: var(--danger);
  font-weight: 700;
}
@media (max-width: 560px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
