<script setup>
import { computed } from 'vue'

const props = defineProps({
  value: { type: Number, default: 0 }, // 0..1
  size: { type: Number, default: 120 },
  stroke: { type: Number, default: 9 },
  color: { type: String, default: 'var(--primary)' },
})

const radius = computed(() => (props.size - props.stroke) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)
const offset = computed(() => circumference.value * (1 - Math.min(1, Math.max(0, props.value))))
</script>

<template>
  <div class="ring" :style="{ width: size + 'px', height: size + 'px' }">
    <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" aria-hidden="true">
      <circle class="track" :cx="size / 2" :cy="size / 2" :r="radius" :stroke-width="stroke" fill="none" />
      <circle
        class="bar"
        :cx="size / 2"
        :cy="size / 2"
        :r="radius"
        :stroke-width="stroke"
        fill="none"
        stroke-linecap="round"
        :stroke="color"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="offset"
      />
    </svg>
    <div class="center"><slot /></div>
  </div>
</template>

<style scoped>
.ring {
  position: relative;
  flex: none;
}
svg {
  transform: rotate(-90deg);
}
.track {
  stroke: var(--bg-soft);
}
.bar {
  transition: stroke-dashoffset 0.9s linear, stroke 0.3s;
}
.center {
  position: absolute;
  inset: 0;
  display: grid;
  place-content: center;
  text-align: center;
}
</style>
