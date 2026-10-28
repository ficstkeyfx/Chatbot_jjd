<script setup>
import { computed, reactive } from 'vue'
import { Minus, Play, Plus, Square, Timer } from '@lucide/vue'
import ProgressRing from './ProgressRing.vue'
import { formatMmSs } from '../composables/formatText'
import { pomodoro, startPomodoro, stopTimer, timersBusy } from '../composables/useTimers'

const form = reactive({ work: 25, rest: 5, cycles: 4 })
const fields = [
  { key: 'work', label: 'Tập trung', unit: 'phút', min: 1, max: 120, step: 5 },
  { key: 'rest', label: 'Nghỉ', unit: 'phút', min: 1, max: 60, step: 1 },
  { key: 'cycles', label: 'Chu kỳ', unit: 'lần', min: 1, max: 12, step: 1 },
]

function bump(field, direction) {
  form[field.key] = Math.min(field.max, Math.max(field.min, form[field.key] + direction * field.step))
}

const focusing = computed(() => pomodoro.value?.phase === 'focus')
</script>

<template>
  <section class="card" aria-labelledby="pomodoro-title">
    <header>
      <span class="badge"><Timer :size="16" /></span>
      <h3 id="pomodoro-title">Pomodoro</h3>
      <span v-if="pomodoro" class="phase" :class="{ rest: !focusing }">
        {{ focusing ? 'Tập trung' : 'Nghỉ ngơi' }}
      </span>
    </header>

    <div v-if="pomodoro" class="running">
      <ProgressRing :value="pomodoro.progress" :size="108" :stroke="8" :color="focusing ? 'var(--primary)' : 'var(--accent)'">
        <span class="time" aria-live="off">{{ formatMmSs(pomodoro.left) }}</span>
        <span class="sub">Chu kỳ {{ pomodoro.cycle }}/{{ pomodoro.total_cycles }}</span>
      </ProgressRing>
      <div class="side">
        <div class="dots" :aria-label="`Chu kỳ ${pomodoro.cycle} trên ${pomodoro.total_cycles}`">
          <i
            v-for="n in pomodoro.total_cycles"
            :key="n"
            :class="{ done: n < pomodoro.cycle, now: n === pomodoro.cycle }"
          />
        </div>
        <p>{{ pomodoro.work_minutes }} phút học · {{ pomodoro.break_minutes }} phút nghỉ</p>
        <button class="btn btn-danger" :disabled="timersBusy" @click="stopTimer('pomodoro')">
          <Square :size="15" /> Dừng
        </button>
      </div>
    </div>

    <div v-else class="idle">
      <div v-for="field in fields" :key="field.key" class="row">
        <span>{{ field.label }}</span>
        <div class="stepper">
          <button
            class="icon-btn"
            :aria-label="`Giảm ${field.label}`"
            :disabled="form[field.key] <= field.min"
            @click="bump(field, -1)"
          >
            <Minus :size="15" />
          </button>
          <output>{{ form[field.key] }} <small>{{ field.unit }}</small></output>
          <button
            class="icon-btn"
            :aria-label="`Tăng ${field.label}`"
            :disabled="form[field.key] >= field.max"
            @click="bump(field, 1)"
          >
            <Plus :size="15" />
          </button>
        </div>
      </div>
      <button
        class="btn btn-primary start"
        :disabled="timersBusy"
        @click="startPomodoro(form.work, form.rest, form.cycles)"
      >
        <Play :size="16" /> Bắt đầu
      </button>
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
  margin-bottom: 12px;
}
.badge {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 9px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
h3 {
  margin: 0;
  font-size: 15px;
  flex: 1;
}
.phase {
  font-size: 12px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 99px;
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.phase.rest {
  background: var(--accent-soft);
  color: var(--accent);
}
.running {
  display: flex;
  align-items: center;
  gap: 16px;
}
.time {
  font-size: 24px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.5px;
}
.sub {
  font-size: 11.5px;
  color: var(--text-soft);
}
.side {
  display: grid;
  gap: 8px;
  min-width: 0;
}
.side p {
  margin: 0;
  font-size: 12.5px;
  color: var(--text-soft);
}
.dots {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.dots i {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--bg-soft);
  border: 1px solid var(--border);
}
.dots i.done {
  background: var(--primary);
  border-color: var(--primary);
}
.dots i.now {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-soft);
}
.idle {
  display: grid;
  gap: 8px;
}
.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
}
.stepper {
  display: flex;
  align-items: center;
  gap: 2px;
}
.stepper .icon-btn {
  width: 30px;
  height: 30px;
  border-radius: 9px;
  background: var(--surface-2);
  border: 1px solid var(--border);
}
output {
  min-width: 64px;
  text-align: center;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
output small {
  font-weight: 400;
  color: var(--muted);
}
.start {
  margin-top: 4px;
}
</style>
