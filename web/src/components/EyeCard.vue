<script setup>
import { Eye } from '@lucide/vue'
import { formatMmSs } from '../composables/formatText'
import { eye, requestNotificationPermission, setEyeReminder, timersBusy } from '../composables/useTimers'

async function toggle() {
  const next = !eye.value
  if (await setEyeReminder(next) && next) requestNotificationPermission()
}
</script>

<template>
  <section class="card" aria-labelledby="eye-title">
    <header>
      <span class="badge"><Eye :size="16" /></span>
      <h3 id="eye-title">Nghỉ mắt 20-20-20</h3>
      <button
        class="switch"
        role="switch"
        :aria-checked="!!eye"
        aria-label="Bật nhắc nghỉ mắt"
        :disabled="timersBusy"
        :class="{ on: eye }"
        @click="toggle"
      >
        <i />
      </button>
    </header>
    <p v-if="!eye" class="hint">
      Cứ mỗi 20 phút nhìn vật cách 6 mét trong 20 giây, chớp mắt chậm và thả lỏng vai.
    </p>
    <div v-else class="running">
      <div class="meter" role="progressbar" :aria-valuenow="Math.round(eye.progress * 100)" aria-valuemin="0" aria-valuemax="100">
        <i :style="{ width: eye.progress * 100 + '%' }" />
      </div>
      <p>
        Nhắc tiếp theo sau <strong>{{ formatMmSs(eye.left) }}</strong>
      </p>
    </div>
  </section>
</template>

<style scoped>
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: var(--shadow-sm);
}
header {
  display: flex;
  align-items: center;
  gap: 10px;
}
.badge {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: var(--info-soft);
  color: #3a86d1;
}
h3 {
  margin: 0;
  font-size: 15px;
  flex: 1;
}
.hint,
.running p {
  margin: 10px 0 0;
  font-size: 12.5px;
  color: var(--text-soft);
}
.running p strong {
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.meter {
  margin-top: 12px;
  height: 8px;
  border-radius: 99px;
  background: var(--bg-soft);
  overflow: hidden;
}
.meter i {
  display: block;
  height: 100%;
  border-radius: 99px;
  background: linear-gradient(90deg, #5aa7ea, var(--primary));
  transition: width 0.9s linear;
}
.switch {
  position: relative;
  width: 44px;
  height: 26px;
  border-radius: 99px;
  background: var(--border);
  transition: background 0.2s;
}
.switch i {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
  transition: transform 0.2s;
}
.switch.on {
  background: var(--primary);
}
.switch.on i {
  transform: translateX(18px);
}
</style>
