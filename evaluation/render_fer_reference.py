"""Create figures and a readable report from measured FER reference predictions."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.runtime/fer-benchmark-deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation.fer_benchmark import metrics, sweep
from fer_model import LABELS


def main():
    out = ROOT / 'outputs/fer-benchmark'
    report_path = out / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    # Supports postprocessing an interrupted/older runner without rerunning inference.
    if 'calibration_split' not in report:
        manifest = json.loads((ROOT / '.runtime/fer2013-benchmark-data/manifest.json').read_text())
        private_hashes = set(manifest['splits']['privateTest']['pixel_hashes'])
        def read_records(name):
            with (out / name).open(encoding='utf-8', newline='') as handle:
                return list(csv.DictReader(handle))
        public = read_records('public_predictions.csv')
        calibration = [(r['truth'], np.array([float(r['p_' + label]) for label in LABELS])) for r in public if manifest['splits']['publicTest']['pixel_hashes'][int(Path(r['image']).stem)] not in private_hashes]
        grid, selected = sweep(calibration, 0.6)
        private = [(r['truth'], np.array([float(r['p_' + label]) for label in LABELS])) for r in read_records('private_predictions.csv')]
        report['validation_threshold_grid'] = grid
        report['public_selected_threshold'] = selected
        report['private_public_selected_threshold'] = metrics(private, selected['min_confidence'], selected['min_margin'])
        report['calibration_split'] = {'name': 'publicTest with exact privateTest overlaps removed', 'count': len(calibration), 'excluded_rows': len(public) - len(calibration)}
        report['calibration_recomputed_from_saved_predictions'] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    raw, app, candidate = (report[k] for k in ('private_raw_top1', 'private_app_threshold', 'private_public_selected_threshold'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.6, 1]}, constrained_layout=True)
    columns = [*LABELS, 'unclear']
    matrix = np.array([[app['confusion'][t][p] for p in columns] for t in LABELS])
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
    heat = axes[0].imshow(normalized, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    axes[0].set_xticks(range(len(columns)), columns, rotation=40, ha='right')
    axes[0].set_yticks(range(len(LABELS)), LABELS)
    axes[0].set(xlabel='Predicted label', ylabel='Dataset label', title='App threshold: confidence >= 0.50, margin >= 0.12')
    for y in range(7):
        for x in range(8):
            axes[0].text(x, y, f'{matrix[y,x]}\n{normalized[y,x]:.0f}%', ha='center', va='center', fontsize=8, color='white' if normalized[y,x] > 55 else '#10243e')
    fig.colorbar(heat, ax=axes[0], shrink=0.65, label='% of true class')
    x = np.arange(7)
    axes[1].barh(x - .18, [raw['per_class'][k]['f1'] for k in LABELS], height=.34, label='Always predict', color='#2474e5')
    axes[1].barh(x + .18, [app['per_class'][k]['f1'] for k in LABELS], height=.34, label='App incl. abstention', color='#169b83')
    axes[1].set_yticks(x, LABELS)
    axes[1].set(xlim=(0, 1), xlabel='F1', title='Per-class F1 on all 3,589 privateTest images')
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower right')
    fig.suptitle('FER2013 reference benchmark — current ONNX model', fontsize=16)
    fig.savefig(out / 'confusion-and-f1.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    grid = report['validation_threshold_grid']
    ax.scatter([r['coverage'] * 100 for r in grid], [r['accuracy_answered'] * 100 for r in grid], s=30, color='#2474e5', alpha=.6)
    current_public = next(r for r in grid if r['min_confidence'] == .5 and r['min_margin'] == .12)
    chosen = report['public_selected_threshold']
    ax.scatter([current_public['coverage'] * 100], [current_public['accuracy_answered'] * 100], marker='*', s=230, color='#e67424', label='Current app threshold')
    ax.scatter([chosen['coverage'] * 100], [chosen['accuracy_answered'] * 100], marker='D', s=80, color='#169b83', label='Candidate (coverage >= 60%)')
    ax.set(xlabel='Coverage (%)', ylabel='Accuracy among answered images (%)', title='Threshold trade-off on public calibration only')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / 'coverage-tradeoff.png', dpi=160)
    plt.close(fig)
    with (out / 'per-class.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['label', 'n', 'raw_precision', 'raw_recall', 'raw_f1', 'app_precision', 'app_recall', 'app_f1'])
        for label in LABELS:
            r, a = raw['per_class'][label], app['per_class'][label]
            writer.writerow([label, r['support'], r['precision'], r['recall'], r['f1'], a['precision'], a['recall'], a['f1']])
    pct = lambda value: f'{100 * value:.2f}%'
    lines = ['# Kết quả benchmark FER thực đo', '', f"Thời điểm UTC: {report['created_utc']}", '',
             'Tập chính: FER2013 privateTest, 3.589 ảnh grayscale 48×48, đủ 7 nhãn. Chỉ đánh giá model đang cài; không fine-tune, không đổi ngưỡng app.', '',
             '| Chế độ | Accuracy toàn tập | Accuracy ảnh được trả lời | Coverage | Macro-F1 toàn tập |',
             '|---|---:|---:|---:|---:|']
    for name, row in [('Luôn chọn nhãn top-1', raw), ('Ngưỡng app hiện tại (0.50 / 0.12)', app), ('Ngưỡng chọn trên public đã bỏ ảnh trùng', candidate)]:
        lines.append(f"| {name} | {pct(row['accuracy_overall'])} | {pct(row['accuracy_answered'])} | {pct(row['coverage'])} | {row['macro_f1']:.4f} |")
    lines += ['', 'Accuracy toàn tập tính unclear là chưa trả lời đúng; macro-F1 cũng tính các lượt từ chối vào số bỏ sót. Accuracy trên ảnh được trả lời chỉ xét phần model chịu đưa nhãn.', '',
              f"Ngưỡng ứng viên: confidence={candidate['min_confidence']}, margin={candidate['min_margin']}; chỉ khảo sát, chưa áp dụng cho app.",
              f"Tập calibration: {report['calibration_split']['count']} ảnh public; loại {report['calibration_split']['excluded_rows']} hàng có nội dung pixel trùng privateTest. Hai tập có {report['dataset']['public_private_exact_overlap']} nội dung ảnh trùng nhau.", '',
              '| Biểu cảm | Số ảnh | F1 top-1 | F1 ngưỡng app |', '|---|---:|---:|---:|']
    for label in LABELS:
        lines.append(f"| {label} | {raw['per_class'][label]['support']} | {raw['per_class'][label]['f1']:.4f} | {app['per_class'][label]['f1']:.4f} |")
    latency = report['latency']['private_cropped']
    lines += ['', f"FER trên CPU, {report['runtime']['threads']} luồng: trung vị {latency['median_ms']:.2f} ms/ảnh; P95 {latency['p95_ms']:.2f} ms. Bao gồm đọc/tiền xử lý ảnh và hai lượt CNN (ảnh gốc + lật), không gồm Qwen hay mạng.", '',
              '## Giới hạn cần giữ khi trích dẫn', '',
              '- Đây là benchmark tham chiếu FER2013. Chưa xác minh được danh sách ảnh từng dùng để train trọng số web-app gốc, nên chưa chứng minh tập này hoàn toàn độc lập với training của tác giả.',
              '- Ảnh có nhãn biểu cảm, không phải nhãn trạng thái tâm lý hay bệnh lý.',
              '- Kết quả trên ảnh mặt 48×48 không chứng minh độ chính xác trên camera người dùng, ảnh tối/nghiêng/che mặt hoặc người Việt.',
              '- Không dùng kết quả privateTest để chọn lại ngưỡng.', '',
              '![Confusion matrix và F1](confusion-and-f1.png)', '', '![Độ phủ và accuracy](coverage-tradeoff.png)', '']
    if 'private_yunet_diagnostic' in report:
        diag = report['private_yunet_diagnostic']
        lines += ['## Chẩn đoán YuNet trên ảnh cực nhỏ', '', f"Trên chính ảnh đã cắt 48×48: coverage {pct(diag['coverage'])}; accuracy toàn tập {pct(diag['accuracy_overall'])}. Số mặt tìm được: {report['detector_face_counts']}.", '',
                  'Phép đo này chỉ kiểm tra việc pipeline từ chối ảnh đầu vào cực nhỏ; KHÔNG phải benchmark ảnh camera đầy đủ và KHÔNG phải độ chính xác bộ dò mặt có bounding-box ground truth.', '']
    lines += ['## Nguồn và tái lập', '', f"Dataset: {report['dataset']['source']}", f"Revision: `{report['dataset']['revision']}`", f"Model SHA256: `{report['model_sha256']}`", '',
              'Từ thư mục project:', '```powershell', '.venv\\Scripts\\python.exe -m evaluation.prepare_fer_reference', '.venv\\Scripts\\python.exe -m evaluation.run_fer_reference --dataset .runtime/fer2013-benchmark-data --output outputs/fer-benchmark --detector-diagnostics', '.venv\\Scripts\\python.exe -m evaluation.render_fer_reference', '```', '',
              'Kết quả chi tiết: report.json, private_predictions.csv, public_predictions.csv, per-class.csv; không kèm ảnh gốc trong thư mục báo cáo.', '']
    (out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'top1_accuracy': raw['accuracy_overall'], 'app_accuracy_answered': app['accuracy_answered'], 'app_coverage': app['coverage'], 'app_macro_f1': app['macro_f1'], 'candidate': [candidate['min_confidence'], candidate['min_margin']]}, indent=2))


if __name__ == '__main__':
    main()
