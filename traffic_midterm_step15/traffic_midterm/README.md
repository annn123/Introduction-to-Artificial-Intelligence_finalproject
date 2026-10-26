# 車禍主張驗證期中原型：第十五步交付版

先閱讀 PROJECT_HISTORY.md，內有目前可重建的完整開發過程、修改原因、標註流程、結果與限制。詳細錯誤分析位於 docs/STEP14_RESULTS_AND_ERRORS.md。

這是目前程式與紀錄的可獨立解壓快照。原有專案已運作時，不必覆蓋或重新安裝。本包不含模型權重，也沒有製作／附入舊版 PDF、TeX 或報告產生器。

## 執行

核心需 Python 3.10+，只用標準函式庫，不需 pip 安裝。以下指令在含 main.py 的資料夾執行。

```cmd
python main.py serve
```

瀏覽器開啟 http://127.0.0.1:8765 。離線規則模式不需要模型。使用本機模型模式前，需自行備妥 Ollama 與 qwen2.5:3b-instruct；已安裝者不必重裝。

```cmd
python main.py check-model --model qwen2.5:3b-instruct
python -m unittest discover -s tests
python tools/verify_saved_results.py
```

最後一個指令只驗證保存檔案，不呼叫模型。

## 本次實際結果

| 條件 | 答對／15 | Accuracy | 四類 Macro-F1 |
| --- | --- | --- | --- |
| LLM Only | 5／15 | 33.3% | 0.350 |
| LLM＋RAG | 7／15 | 46.7% | 0.500 |
| Verification 最終 | 15／15 | 100% | 1.000 |
| 純規則（第十五步事後診斷） | 15／15 | 100% | 1.000 |

Verification 組模型原始標籤答對 9／15，六項由規則修正。最終標籤可由純規則完全重現，因此不能把 100% 當作模型推理能力或 LLM 對最終分類的額外貢獻。

資料為 10 個合成測試案例、15 項固定主張。沒有部分支持案例。兩位組員初始答案 11 項一致、4 項分歧；最終分歧及來源選集含 AI 協助，由使用者採用。原始標註及確認流程均保留，不稱為純人工共識 gold。部分情境與開發重疊。

## 重現比較

原始結果在 results/step13_comparison，不必為交付重新跑模型。如需新一輪，另存輸出資料夾：

```cmd
python main.py evaluate --backend ollama --model qwen2.5:3b-instruct --split test --gold data/reference_annotations.json --output results/reproduction_new_run
```

純規則重現不需 Ollama：

```cmd
python main.py evaluate --backend offline --split test --gold data/reference_annotations.json --output results/rules_reproduction_new_run
```

## 檔案導覽

- traffic/、main.py、static/：目前核心與本機介面。
- tests/：153 項工程測試，包含模擬模型，不是模型效果評分。
- data/corpus.json：12 筆法規、10 筆相似裁判摘要。
- data/cases.json：5 件開發案例及 10 件測試案例；含原始 silver labels，正式本次比較改以 reference_annotations.json 為依據。
- data/reference_annotations.json：已採用參考標籤、來源 ID、原始雙人標註與實際確認方式。
- results/step11_dev_pilot/：開發組原始試跑。
- results/step13_comparison/：使用者實機執行的三組原始 JSON。
- results/step15_rules_baseline/：本步執行的純規則診斷。
- records/annotation_initial.xlsx：使用者回傳的原始雙人標註。
- records/development_exports/：先前開發案例匯出及模型失敗紀錄，不是獨立測試。
- records/update_notes/：過去更新說明，屬歷史紀錄，版本與測試數可能早於本次。
- records/source_receipts_historical.json：歷史網址可達性紀錄，不代表本次重查或法律內容審定。
- records/verification_receipt.json：本步驗證及檔案版本紀錄。
- MANIFEST.json：交付包內檔案的 SHA-256；不含自身。

尚未量測擷取 F1、人工幻覺率、引用語義支持率及調解品質。調解為模板，證據為使用者提供的結構化轉錄，不會自動檢驗真實影片或判責比例。核心只連本機 Ollama；沒有付費推論 API。
