<script setup>
import { computed, ref } from 'vue'
import { Check, Copy, FileUp, Loader, Sparkles, X } from '@lucide/vue'
import ModalShell from './ModalShell.vue'
import { formatText } from '../composables/formatText'
import { summarize } from '../composables/useChat'
import { toast } from '../composables/useToast'
import { closeDialog } from '../composables/useUi'

const MAX_CHARS = 50000
const MAX_BYTES = 10 * 1024 * 1024

const tab = ref('text')
const text = ref('')
const file = ref(null)
const result = ref('')
const loading = ref(false)
const copied = ref(false)
const dragging = ref(false)
const fileInput = ref(null)
let controller = null

const canRun = computed(() => !loading.value && (tab.value === 'text' ? text.value.trim() : file.value))
const blocks = computed(() => formatText(result.value))

function pickFile(candidate) {
  if (!candidate) return
  if (candidate.size > MAX_BYTES) {
    toast('Tệp vượt quá 10 MB.', { type: 'error' })
    return
  }
  file.value = candidate
}

function onDrop(event) {
  dragging.value = false
  pickFile(event.dataTransfer?.files?.[0])
}

async function run() {
  if (!canRun.value) return
  loading.value = true
  result.value = ''
  controller = new AbortController()
  try {
    result.value = await summarize(
      tab.value === 'text' ? { text: text.value } : { file: file.value },
      controller.signal,
    )
  } catch (error) {
    if (error.name !== 'AbortError') toast(error.message, { type: 'error' })
  } finally {
    loading.value = false
    controller = null
  }
}

function cancel() {
  controller?.abort()
}

async function copy() {
  try {
    await navigator.clipboard.writeText(result.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch {
    toast('Không sao chép được nội dung.', { type: 'error' })
  }
}

function fileSize(bytes) {
  return bytes > 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`
}
</script>

<template>
  <ModalShell
    title="Tóm tắt bài học"
    subtitle="Tối đa 3 ý chính và một câu cần nhớ, giúp bạn đỡ phải đọc dài trên màn hình."
    wide
    @close="closeDialog"
  >
    <div class="tabs" role="tablist">
      <button role="tab" :aria-selected="tab === 'text'" :class="{ on: tab === 'text' }" @click="tab = 'text'">
        Dán văn bản
      </button>
      <button role="tab" :aria-selected="tab === 'file'" :class="{ on: tab === 'file' }" @click="tab = 'file'">
        Tải tệp lên
      </button>
    </div>

    <div v-if="tab === 'text'" class="pane">
      <textarea
        v-model="text"
        class="field"
        rows="9"
        :maxlength="MAX_CHARS"
        placeholder="Dán nội dung bài học, ghi chú hoặc tài liệu cần tóm tắt…"
        aria-label="Nội dung cần tóm tắt"
      />
      <small class="count">{{ text.length.toLocaleString('vi-VN') }} / {{ MAX_CHARS.toLocaleString('vi-VN') }} ký tự</small>
    </div>

    <div v-else class="pane">
      <div
        class="drop"
        :class="{ over: dragging, filled: file }"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="onDrop"
        @click="fileInput.click()"
        @keydown.enter="fileInput.click()"
        tabindex="0"
        role="button"
        aria-label="Chọn tệp để tóm tắt"
      >
        <FileUp :size="28" />
        <template v-if="file">
          <strong>{{ file.name }}</strong>
          <small>{{ fileSize(file.size) }}</small>
          <button class="btn btn-ghost" @click.stop="file = null; fileInput.value = ''"><X :size="15" /> Bỏ tệp</button>
        </template>
        <template v-else>
          <strong>Kéo thả hoặc bấm để chọn tệp</strong>
          <small>PDF, DOCX, ảnh có chữ hoặc văn bản (tối đa 10 MB, PDF tối đa 20 trang)</small>
        </template>
      </div>
      <input
        ref="fileInput"
        type="file"
        hidden
        accept=".pdf,.docx,.txt,.md,image/*"
        @change="pickFile($event.target.files?.[0])"
      />
    </div>

    <div class="actions">
      <button v-if="loading" class="btn btn-danger" @click="cancel"><X :size="16" /> Hủy</button>
      <button class="btn btn-primary" :disabled="!canRun" @click="run">
        <Loader v-if="loading" :size="16" class="spin" />
        <Sparkles v-else :size="16" />
        {{ loading ? 'Đang tóm tắt…' : 'Tóm tắt' }}
      </button>
    </div>

    <section v-if="result" class="result" aria-live="polite">
      <header>
        <strong>Kết quả</strong>
        <button class="btn btn-soft" @click="copy">
          <Check v-if="copied" :size="15" />
          <Copy v-else :size="15" />
          {{ copied ? 'Đã chép' : 'Sao chép' }}
        </button>
      </header>
      <template v-for="(block, index) in blocks" :key="index">
        <p v-if="block.type === 'p'" :class="{ remember: block.parts[0]?.text.startsWith('Cần nhớ') }">
          <template v-for="(part, i) in block.parts" :key="i">
            <strong v-if="part.bold">{{ part.text }}</strong>
            <template v-else>{{ part.text }}</template>
          </template>
        </p>
        <component :is="block.type" v-else>
          <li v-for="(item, n) in block.items" :key="n">
            <template v-for="(part, i) in item" :key="i">{{ part.text }}</template>
          </li>
        </component>
      </template>
    </section>
  </ModalShell>
</template>

<style scoped>
.tabs {
  display: flex;
  gap: 4px;
  padding: 4px;
  margin-bottom: 12px;
  background: var(--bg-soft);
  border-radius: 12px;
}
.tabs button {
  flex: 1;
  padding: 8px;
  border-radius: 9px;
  font-weight: 600;
  color: var(--text-soft);
}
.tabs button.on {
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-sm);
}
.pane {
  display: grid;
  gap: 6px;
}
textarea {
  resize: vertical;
  min-height: 160px;
  line-height: 1.5;
}
.count {
  justify-self: end;
  color: var(--muted);
}
.drop {
  display: grid;
  justify-items: center;
  gap: 4px;
  padding: 28px 16px;
  text-align: center;
  border: 2px dashed var(--border);
  border-radius: 16px;
  color: var(--text-soft);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
.drop:hover,
.drop.over {
  border-color: var(--primary);
  background: var(--primary-soft);
}
.drop.filled {
  border-style: solid;
  border-color: var(--primary);
}
.drop strong {
  color: var(--text);
  overflow-wrap: anywhere;
}
.drop small {
  color: var(--muted);
}
.actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 14px;
}
.result {
  margin-top: 18px;
  padding: 16px 18px;
  background: var(--primary-soft);
  border-radius: 16px;
}
.result header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.result p {
  margin: 6px 0;
}
.result .remember {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: 12px;
  background: var(--surface);
  border-left: 4px solid var(--primary);
  font-weight: 600;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
