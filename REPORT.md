# Báo cáo Lab: Giai đoạn 2, Track 3, Day 17 — Memory Systems for AI Agent

**Sinh viên:** LeThanhDat — 2A202600756  
**Ngày hoàn thành:** 2026-06-19

---

## 1. Tổng quan bài làm

Bài lab yêu cầu xây dựng và so sánh hai AI agent với các lớp memory khác nhau:

| Agent | Memory layers |
|---|---|
| **Baseline** | Within-session only (quên sạch khi sang thread mới) |
| **Advanced** | Short-term + `User.md` persistent + Compact memory |

Toàn bộ bài chạy **offline** (không cần API key) bằng heuristic engine. Khi có API key, cả hai agent tự động chuyển sang LLM thật qua `model_provider.py`.

---

## 2. Cấu trúc triển khai

```
src/
  model_provider.py   — build_chat_model() cho 6 providers
  config.py           — LabConfig + load_config() từ env vars
  memory_store.py     — estimate_tokens, UserProfileStore, CompactMemoryManager
  agent_baseline.py   — BaselineAgent (within-session only)
  agent_advanced.py   — AdvancedAgent (3-layer memory)
  benchmark.py        — Standard + Stress benchmark, format_rows()
  test_agents.py      — 13 pytest tests
```

### Provider hỗ trợ

`openai` | `anthropic` | `gemini` | `ollama` | `openrouter` | `custom`

Cấu hình qua biến môi trường trong file `.env` (xem `.env.example`).

---

## 3. Kết quả Benchmark

### 3.1 Standard Benchmark (`data/conversations.json`)

10 cuộc hội thoại, 10 turns mỗi cuộc, 14 recall questions.

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 1 547             | 14 907                  | 0.000                | 0.100            | 0                     | 0           |
| Advanced | 1 403             | 20 910                  | **0.536**            | **0.607**        | 2 855                 | 0           |

### 3.2 Long-Context Stress Benchmark (`data/advanced_long_context.json`)

1 cuộc hội thoại rất dài (16 turns dày đặc), 3 recall questions.

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|:-----------------:|:-----------------------:|:--------------------:|:----------------:|:---------------------:|:-----------:|
| Baseline | 188               | 21 846                  | 0.000                | 0.100            | 0                     | 0           |
| Advanced | 255               | **9 731**               | **0.667**            | **0.733**        | 279                   | 5           |

---

## 4. Phân tích kết quả

### 4.1 Tại sao Advanced có recall tốt hơn Baseline?

Baseline chỉ có **within-session memory**: khi sang thread mới để hỏi recall questions, nó không nhớ bất kỳ thông tin nào đã trao đổi. Recall = 0.000 là kết quả tất yếu.

Advanced lưu facts ổn định vào **`User.md` bền vững**: tên, nơi ở, nghề nghiệp, đồ uống yêu thích, style trả lời. Khi sang thread mới, agent đọc lại file này và trả lời đúng recall questions. Kết quả:
- Standard: recall = **0.536** (53.6% câu hỏi được trả lời đúng ít nhất 50%)
- Stress: recall = **0.667**

### 4.2 Tại sao Advanced có thể tốn hơn Baseline ở hội thoại ngắn?

Ở standard benchmark:
- Baseline `prompt_tokens_processed` = 14 907
- Advanced `prompt_tokens_processed` = 20 910 (+40%)

Lý do: mỗi lượt hội thoại, Advanced phải inject toàn bộ `User.md` + compact summary vào context, ngay cả khi hội thoại còn rất ngắn. Đây là chi phí overhead cố định của persistent memory.

Baseline chỉ mang theo messages trong session hiện tại — với hội thoại ngắn, nó "rẻ" hơn. **Trade-off rõ ràng: tiết kiệm token nhưng mất khả năng nhớ dài hạn.**

### 4.3 Tại sao Compact Memory giúp Advanced ở hội thoại dài?

Ở stress benchmark:
- Baseline: `prompt_tokens_processed` = 21 846 — tăng theo kiểu O(n²) vì mỗi lượt phải kéo toàn bộ lịch sử.
- Advanced: `prompt_tokens_processed` = **9 731** (giảm 55%) với **5 lần compaction**.

Cơ chế: khi tổng tokens vượt `compact_threshold_tokens`, `CompactMemoryManager` giữ lại `keep_messages` gần nhất và nén messages cũ thành summary ngắn (`_SUMMARY_SNIPPET_CHARS = 60` chars/message, cap tối đa `_SUMMARY_MAX_TOKENS = 200` tokens). Context không phình to vô hạn.

**Compact chủ yếu tối ưu `prompt tokens processed`, không phải `agent tokens only`** — số token generate ra của Advanced (255) chỉ nhỉnh hơn Baseline (188) một chút vì câu trả lời dài hơn do có context tốt hơn.

### 4.4 Memory file tăng trưởng như thế nào và rủi ro gì?

`User.md` tăng theo số facts được extract qua các cuộc hội thoại:
- Standard (10 convs × 10 turns): **2 855 bytes**
- Stress (1 conv × 16 turns dài): **279 bytes** — ít hơn vì nội dung là news/analysis, không phải personal info

**Rủi ro thực tế:**

| Rủi ro | Mô tả | Giải pháp đã áp dụng |
|---|---|---|
| Lưu sai fact | Capture câu hỏi nhầm thành fact ("đồ uống yêu thích của mình là gì" → ghi vào DB) | Confidence threshold + noise value filter |
| Correction bị bỏ qua | "không còn là X" bị interpret là X mới | Negation-aware extraction, `is_correction_message()` |
| File phình to | Không có cơ chế xóa facts cũ | `upsert_fact` REPLACE thay vì append; decay score ưu tiên facts mới |
| Facts hết hạn | Địa chỉ cũ 6 tháng trước vẫn được dùng như hiện tại | `decay_score()` giảm độ ưu tiên theo tuổi |

---

## 5. Bonus Features

### 5.1 Confidence Threshold

**Vấn đề**: Mọi regex match đều được ghi vào `User.md`, kể cả khi agent không chắc.

**Giải pháp**: Hàm `_score_fact(key, message, value)` tính confidence 0.0–1.0:

| Tình huống | Score |
|---|---|
| Explicit correction ("đính chính", "chuyển sang") | 0.92 |
| Direct assertion ("mình tên là", "hiện đang là", "mình đang ở") | 0.85 |
| Clear statement, không có dấu "?" | 0.72 |
| Ambiguous / ngắn / không rõ | 0.50 |

`CONFIDENCE_THRESHOLD = 0.65` — chỉ facts có score >= 0.65 được persist vào `User.md`.

**Cải thiện**: Giảm false positives đáng kể (từ "đồ uống yêu thích của mình là gì" → không được ghi).  
**Rủi ro**: False negatives khi người dùng diễn đạt gián tiếp hoặc ngắn gọn.

**Test liên quan:**
```
test_confidence_threshold_blocks_questions  — câu hỏi không được ghi
test_confidence_threshold_accepts_assertions — assertion trực tiếp được ghi
```

### 5.2 Conflict Handling

**Vấn đề**: Khi người dùng đính chính ("không còn là backend engineer, giờ MLOps engineer"), agent có thể giữ cả 2 giá trị hoặc cập nhật sai.

**Giải pháp** — 3 lớp bảo vệ:

1. **`is_correction_message(message)`**: detect từ khóa correction trong toàn bộ message → boost confidence +0.1 cho tất cả facts trong lượt đó.

2. **Negation-aware extraction**: khi tìm profession, kiểm tra 50 ký tự trước match xem có chứa "không còn / không phải / chứ không" không. Nếu có → bỏ qua match đó (đây là fact cũ đang bị phủ định).

3. **Priority-based profession extraction**: tìm "chuyển sang X" và "giờ làm X" trước; chỉ fallback về pattern chung nếu không tìm thấy.

4. **`upsert_fact` REPLACE**: không bao giờ có 2 dòng cho cùng 1 key trong `User.md`.

**Kết quả**: Sau conv-06 ("giờ chuyển sang MLOps engineer"), `User.md` chứa `MLOps engineer`, không phải `backend engineer`.

**Test liên quan:**
```
test_conflict_handling_overwrites_old_fact  — overwrite không tạo duplicate
test_conflict_correction_via_agent          — agent cập nhật đúng sau correction
```

### 5.3 Memory Decay

**Vấn đề**: Một fact từ 6 tháng trước ít liên quan hơn fact từ hôm qua, nhưng không có cơ chế phân biệt.

**Giải pháp**: Mỗi fact lưu inline metadata:
```
- nơi ở: Huế [c:0.85 t:2026-06-19 m:5]
           ^confidence  ^last_update  ^mention_count
```

Công thức `decay_score(user_id, key)`:
```python
decay_rate = max(0.01, 0.05 / sqrt(mentions))  # nhiều lần nhắc → decay chậm
score = max(0.1, 1.0 - age_days * decay_rate)
```

- Fact được nhắc 1 lần: giảm 5%/ngày
- Fact được nhắc 9 lần: giảm 1.67%/ngày
- Score tối thiểu: 0.1 (không bao giờ bị xóa hoàn toàn)

`prioritized_facts(user_id)` → danh sách `(key, value, decay_score)` sắp xếp giảm dần. Agent có thể dùng để ưu tiên facts khi context bị giới hạn.

**Cải thiện**: Tên và nghề nghiệp (nhắc nhiều) decay chậm; địa điểm tạm thời (nhắc ít) decay nhanh.  
**Rủi ro**: Cần calibrate `decay_rate` theo domain. Facts cũ không bị xóa, chỉ giảm priority.

**Test liên quan:**
```
test_decay_score_recent_fact_is_high   — fact hôm nay score >= 0.9
test_decay_score_missing_fact_is_zero  — fact chưa ghi score = 0.0
test_prioritized_facts_order           — fact mới rank cao hơn fact cũ 2020
```

---

## 6. Kết quả Test

```
13 passed in 0.17s
```

| Test | Mô tả |
|---|---|
| `test_user_markdown_read_write_edit` | User.md CRUD hoạt động đúng |
| `test_upsert_fact_insert_and_update` | upsert không tạo duplicate |
| `test_compact_trigger` | Compact kích hoạt khi vượt threshold |
| `test_compact_does_not_trigger_on_short_thread` | Không compact sớm |
| `test_cross_session_recall` | Advanced nhớ, Baseline quên cross-session |
| `test_compact_reduces_prompt_load_on_long_thread` | Advanced < Baseline về prompt tokens ở thread dài |
| `test_confidence_threshold_blocks_questions` | Câu hỏi không được extract |
| `test_confidence_threshold_accepts_assertions` | Assertion trực tiếp được extract |
| `test_conflict_handling_overwrites_old_fact` | Conflict → overwrite đúng |
| `test_conflict_correction_via_agent` | Agent xử lý correction đúng |
| `test_decay_score_recent_fact_is_high` | Fact mới score cao |
| `test_decay_score_missing_fact_is_zero` | Fact chưa ghi score = 0 |
| `test_prioritized_facts_order` | Sắp xếp theo recency đúng |

---

## 7. Cách chạy

```bash
# 1. Setup môi trường
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install langchain langgraph langchain-openai langchain-google-genai \
            langchain-anthropic langchain-ollama langchain-openrouter \
            python-dotenv tabulate pytest

# 2. Chạy toàn bộ tests (offline, không cần API key)
python -m pytest -v

# 3. Chạy benchmark (offline, không cần API key)
cd src
python benchmark.py

# 4. Chạy với LLM thật
cp .env.example .env
# Mở .env, điền API key vào OPENAI_API_KEY hoặc provider tương ứng
python benchmark.py
```

---

## 8. Kết luận

Bài lab đã chứng minh rõ luồng logic:

1. **Baseline không nhớ dài hạn** — recall = 0.000 cross-session, chi phí context thấp.
2. **Advanced thêm `User.md`** — recall tăng lên 0.536–0.667, nhưng overhead per-turn cao hơn.
3. **Hội thoại dài làm prompt cost Baseline tăng mạnh** — O(n²) không giới hạn.
4. **Compact memory kéo chi phí ngữ cảnh của Advanced xuống** — tiết kiệm 55% prompt tokens ở stress test với 5 compaction.
5. **Hệ thống Advanced mạnh hơn nhưng cần guardrail** — confidence threshold, conflict handling, và memory decay là các cơ chế cần thiết để tránh lưu sai hoặc lưu thông tin cũ.

Bonus features (confidence threshold, conflict handling, memory decay) giải quyết các rủi ro thực tế của persistent memory, làm cho hệ thống đáng tin cậy hơn trong production.
