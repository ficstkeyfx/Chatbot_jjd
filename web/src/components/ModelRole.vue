<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { CircleAlert, CircleCheck, KeyRound, Loader, Plug, Save, Trash2 } from '@lucide/vue'
import { api } from '../composables/api'
import { loadLocal, local } from '../composables/useLocal'
import { providerById, saveSettings, settings } from '../composables/useSettings'

const props = defineProps({
  role: { type: String, required: true }, // llm | vision | stt | tts
  title: { type: String, required: true },
  hint: { type: String, default: '' },
})

const draft = reactive({ provider: '', model: '', base_url: '', voice: '' })
const baseline = reactive({ provider: '', model: '', base_url: '', voice: '' })
const keyInput = ref('')
const showKey = ref(false)
const saving = ref(false)
const testing = ref(false)
const result = ref(null)

function sync() {
  const server = settings.profile[props.role] || {}
  const next = {
    provider: server.provider || '',
    model: server.model || '',
    base_url: server.base_url || '',
    voice: server.voice || '',
  }
  const untouched = JSON.stringify({ ...draft }) === JSON.stringify({ ...baseline })
  Object.assign(baseline, next)
  if (untouched) Object.assign(draft, next) // không ghi đè khi người dùng đang sửa dở
}
watch(() => settings.profile[props.role], sync, { immediate: true, deep: true })

const choices = computed(() => [
  ...Object.entries(settings.special[props.role] || {}).map(([id, label]) => ({ id, label })),
  ...settings.providers.filter((item) => item.roles.includes(props.role)).map(({ id, label }) => ({ id, label })),
])
const catalog = computed(() => providerById(draft.provider))
const isSpecial = computed(() => draft.provider in (settings.special[props.role] || {}))
const isLocal = computed(() => draft.provider === 'local')
const isCustom = computed(() => !!catalog.value?.custom_url)
const usesKey = computed(() => !!catalog.value && !isLocal.value)
const keyRequired = computed(() => !!catalog.value?.needs_key)

const keyState = computed(() => {
  if (settings.keys[draft.provider]) return 'saved'
  if (settings.server_keys[draft.provider]) return 'server'
  return 'missing'
})

const suggestions = computed(() => {
  const items = [...(catalog.value?.suggestions?.[props.role] || [])]
  if (isLocal.value && (props.role === 'llm' || props.role === 'vision')) items.unshift(...local.llm_models)
  return [...new Set(items)]
})
const voices = computed(() => {
  if (props.role !== 'tts') return []
  if (isLocal.value) return [...new Set([...local.voices.installed, ...local.voices.downloadable])]
  return catalog.value?.voices || []
})
const placeholder = computed(() => {
  const sameProvider = draft.provider === baseline.provider
  if (sameProvider && settings.effective[props.role]) return settings.effective[props.role]
  return catalog.value?.models?.[props.role] || 'mặc định của nhà cung cấp'
})

const dirty = computed(
  () =>
    !!keyInput.value ||
    ['provider', 'model', 'base_url', 'voice'].some((field) => (draft[field] || '') !== (baseline[field] || '')),
)
const problem = computed(() => {
  if (isSpecial.value || !catalog.value) return ''
  if (keyRequired.value && keyState.value === 'missing' && !keyInput.value) {
    return `Nhập API key ${catalog.value.label} để dùng nhà cung cấp này.`
  }
  if (isCustom.value && (!draft.base_url || !draft.model)) return 'Nhà cung cấp tùy chỉnh cần địa chỉ API và tên model.'
  return ''
})
const canSave = computed(() => dirty.value && !problem.value && !saving.value)

function pickProvider() {
  // Đổi nhà cung cấp: model/giọng cũ không còn ý nghĩa.
  draft.model = ''
  draft.voice = ''
  draft.base_url = ''
  keyInput.value = ''
  result.value = null
  if (draft.provider === 'local') loadLocal()
}

async function save() {
  saving.value = true
  const entry = { provider: draft.provider }
  if (!isSpecial.value) {
    entry.model = draft.model.trim()
    if (isCustom.value) entry.base_url = draft.base_url.trim()
    if (props.role === 'tts') entry.voice = draft.voice.trim()
  }
  const patch = { [props.role]: entry }
  if (keyInput.value && usesKey.value) patch.api_keys = { [draft.provider]: keyInput.value.trim() }
  if (await saveSettings(patch)) {
    keyInput.value = ''
    result.value = null
  }
  saving.value = false
}

async function removeKey() {
  await saveSettings({ api_keys: { [draft.provider]: null } })
}

async function test() {
  testing.value = true
  result.value = null
  try {
    result.value = await api.testRole(props.role)
  } catch (error) {
    result.value = { ok: false, detail: error.message }
  } finally {
    testing.value = false
  }
}

onMounted(() => {
  if (isLocal.value) loadLocal()
})
</script>

<template>
  <section class="role">
    <header>
      <div>
        <h3>{{ title }}</h3>
        <p v-if="hint">{{ hint }}</p>
      </div>
      <span v-if="settings.effective[role]" class="chip" :title="`Đang dùng: ${settings.effective[role]}`">
        {{ settings.effective[role] }}
      </span>
    </header>

    <div class="grid">
      <label>
        Nhà cung cấp
        <select v-model="draft.provider" class="field" @change="pickProvider">
          <option v-for="choice in choices" :key="choice.id" :value="choice.id">{{ choice.label }}</option>
        </select>
      </label>

      <label v-if="!isSpecial">
        Tên model
        <input
          v-model="draft.model"
          class="field"
          spellcheck="false"
          autocomplete="off"
          :list="`models-${role}`"
          :placeholder="placeholder"
        />
        <datalist :id="`models-${role}`">
          <option v-for="item in suggestions" :key="item" :value="item" />
        </datalist>
      </label>

      <label v-if="role === 'tts' && !isSpecial">
        Giọng đọc
        <input
          v-model="draft.voice"
          class="field"
          spellcheck="false"
          autocomplete="off"
          :list="`voices-${role}`"
          placeholder="mặc định"
        />
        <datalist :id="`voices-${role}`">
          <option v-for="item in voices" :key="item" :value="item" />
        </datalist>
      </label>

      <label v-if="isCustom" class="wide">
        Địa chỉ API (gốc, ví dụ https://host/v1)
        <input v-model="draft.base_url" class="field" spellcheck="false" inputmode="url" placeholder="https://" />
      </label>
    </div>

    <p v-if="catalog?.note" class="note">{{ catalog.note }}</p>
    <p v-if="isLocal && role !== 'stt' && role !== 'tts' && local.loaded && !local.llm_available" class="note warn">
      Chưa thấy dịch vụ Ollama trên máy chủ. Hãy chọn nhà cung cấp khác hoặc nhờ quản trị viên cài đặt.
    </p>

    <div v-if="usesKey" class="key">
      <div class="state" :class="keyState">
        <KeyRound :size="15" />
        <span v-if="keyState === 'saved'">Đã lưu API key của bạn</span>
        <span v-else-if="keyState === 'server'">Đang dùng key chung của máy chủ</span>
        <span v-else>{{ keyRequired ? 'Chưa có API key' : 'API key (không bắt buộc)' }}</span>
        <button v-if="keyState === 'saved'" class="btn btn-ghost mini" type="button" @click="removeKey">
          <Trash2 :size="14" /> Xóa key
        </button>
      </div>
      <input
        v-model="keyInput"
        class="field"
        :type="showKey ? 'text' : 'password'"
        autocomplete="off"
        spellcheck="false"
        :placeholder="keyState === 'saved' ? 'Nhập key mới để thay thế' : 'Dán API key vào đây'"
        @focus="showKey = false"
      />
    </div>

    <p v-if="problem && dirty" class="note warn">{{ problem }}</p>

    <footer>
      <p v-if="result" class="result" :class="{ ok: result.ok }" role="status">
        <CircleCheck v-if="result.ok" :size="16" />
        <CircleAlert v-else :size="16" />
        {{ result.detail }}
      </p>
      <span class="spacer" />
      <button class="btn btn-ghost" type="button" :disabled="testing || dirty || isSpecial" :title="dirty ? 'Hãy lưu trước khi kiểm tra' : ''" @click="test">
        <Loader v-if="testing" :size="15" class="spin" /> <Plug v-else :size="15" /> Kiểm tra kết nối
      </button>
      <button class="btn btn-primary" type="button" :disabled="!canSave" @click="save">
        <Loader v-if="saving" :size="15" class="spin" /> <Save v-else :size="15" /> Lưu
      </button>
    </footer>
  </section>
</template>

<style scoped>
.role {
  display: grid;
  gap: 12px;
  padding: 16px;
  margin-bottom: 14px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 16px;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 10px;
}
h3 {
  margin: 0;
  font-size: 15px;
}
header p {
  margin: 2px 0 0;
  font-size: 13px;
  color: var(--text-soft);
}
.chip {
  max-width: 45%;
  padding: 3px 10px;
  border-radius: 99px;
  background: var(--primary-soft);
  color: var(--primary-strong);
  font-size: 12px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 12px;
}
.wide {
  grid-column: 1 / -1;
}
label {
  display: grid;
  gap: 5px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-soft);
}
.note {
  margin: 0;
  font-size: 12.5px;
  color: var(--text-soft);
}
.note.warn {
  color: var(--accent);
}
.key {
  display: grid;
  gap: 8px;
}
.state {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-soft);
}
.state.saved,
.state.server {
  color: var(--primary-strong);
}
.state .mini {
  margin-left: auto;
  padding: 4px 8px;
  font-size: 12.5px;
}
footer {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.spacer {
  flex: 1;
}
.result {
  flex: 1 1 100%;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 0;
  padding: 8px 12px;
  border-radius: 10px;
  background: var(--danger-soft);
  color: var(--danger);
  font-size: 13px;
  overflow-wrap: anywhere;
}
.result.ok {
  background: var(--primary-soft);
  color: var(--primary-strong);
}
.result svg {
  flex: none;
  margin-top: 2px;
}
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (max-width: 560px) {
  .grid {
    grid-template-columns: 1fr;
  }
  .chip {
    display: none;
  }
}
</style>
