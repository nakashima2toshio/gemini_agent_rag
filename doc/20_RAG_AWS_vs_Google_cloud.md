# AWS vs Google Cloud 比較検討書
## RAGシステム移行のための評価レポート

**作成日:** 2025/12/05
**対象システム:** RAGシステム（Python, Streamlit, Redis, Qdrant, Docker-compose, Gemini API, OpenAI API）
**現環境:** AWS EC2
**検討対象:** Google Cloud Platform (GCP)

---

## 目次

1. [エグゼクティブサマリー](#1-エグゼクティブサマリー)
2. [対象システム構成](#2-対象システム構成)
3. [コンピュート比較](#3-コンピュート比較)
4. [Redis（キャッシュ）比較](#4-redisキャッシュ比較)
5. [コンテナ/Docker対応比較](#5-コンテナdocker対応比較)
6. [AI/ML API連携比較](#6-aiml-api連携比較)
7. [総合コスト比較](#7-総合コスト比較)
8. [使いやすさ・開発体験比較](#8-使いやすさ開発体験比較)
9. [移行シナリオ別推奨](#9-移行シナリオ別推奨)
10. [結論と推奨](#10-結論と推奨)

---

## 1. エグゼクティブサマリー

### 結論

| 評価項目 | AWS | GCP | 推奨 |
|----------|-----|-----|------|
| **コスト（VM）** | △ | ◎ | **GCP**（約15%安い） |
| **コスト（Redis）** | ◎ | ○ | **AWS**（ElastiCacheが安い） |
| **Gemini API連携** | △ | ◎ | **GCP**（ネイティブ連携） |
| **サーバーレス** | ○ | ◎ | **GCP**（Cloud Run優秀） |
| **エコシステム成熟度** | ◎ | ○ | **AWS**（サービス数豊富） |
| **学習コスト** | △ | ◎ | **GCP**（シンプル） |
| **日本リージョン** | ◎ | ◎ | 同等 |

### 推奨パターン

```
【Gemini API中心のRAGシステム】→ GCP推奨
  理由: Gemini APIとのネイティブ連携、Cloud Runのサーバーレス、約15%のコスト削減

【既存AWSリソース活用】→ AWS継続推奨
  理由: 移行コスト回避、既存ノウハウ活用

【ハイブリッド構成】→ AWS + Gemini API直接利用
  理由: 移行なしでGemini APIの恩恵を受けられる
```

---

## 2. 対象システム構成

### 現在のRAGシステム構成

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG System Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Streamlit  │────▶│   Python     │────▶│  Gemini API  │    │
│  │   Frontend   │     │   Backend    │     │  OpenAI API  │    │
│  └──────────────┘     └──────┬───────┘     └──────────────┘    │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                        │
│              ┌──────────┐        ┌──────────┐                  │
│              │  Redis   │        │  Qdrant  │                  │
│              │ (Cache)  │        │(VectorDB)│                  │
│              └──────────┘        └──────────┘                  │
│                                                                  │
│              【docker-compose で統合管理】                       │
└─────────────────────────────────────────────────────────────────┘
```

### 対応するクラウドサービス

| コンポーネント | AWS | GCP |
|---------------|-----|-----|
| コンピュート | EC2 | Compute Engine / Cloud Run |
| Redis | ElastiCache | Memorystore |
| Qdrant | EC2上でセルフホスト | GCE上でセルフホスト |
| コンテナ管理 | ECS / EKS | Cloud Run / GKE |
| AI API | Bedrock + 外部API | Vertex AI + Gemini API |

---

## 3. コンピュート比較

### 3.1 VM価格比較（月額概算）

| スペック | AWS EC2 | GCP Compute Engine | 差額 |
|----------|---------|-------------------|------|
| **小規模（2vCPU, 4GB）** | | | |
| t4g.medium | $28.20/月 | - | - |
| e2-medium | - | $24.46/月 | **-13%** |
| **中規模（4vCPU, 16GB）** | | | |
| t4g.xlarge | $56.54/月 | - | - |
| e2-standard-4 | - | $48.55/月 | **-14%** |
| **大規模（8vCPU, 32GB）** | | | |
| m6i.2xlarge | $277.44/月 | - | - |
| e2-standard-8 | - | $233.02/月 | **-16%** |

> **結論**: GCP Compute Engineは一般的に**約15%安価**

### 3.2 割引オプション比較

| 割引タイプ | AWS | GCP |
|-----------|-----|-----|
| **長期契約** | Reserved Instances（最大75%オフ） | Committed Use Discounts（最大57%オフ） |
| **スポット/プリエンプティブル** | Spot Instances（最大90%オフ） | Spot VMs（最大91%オフ） |
| **自動割引** | なし | Sustained Use Discounts（自動最大30%オフ） |

> **ポイント**: GCPは「Sustained Use Discounts」により、長期契約なしでも自動的に割引が適用される点が有利

### 3.3 RAGシステム推奨構成

```
【開発/テスト環境】
AWS: t4g.small ($14.98/月)
GCP: e2-small  ($12.23/月) ← 約18%安い

【本番環境（小〜中規模）】
AWS: t4g.large + ElastiCache ($80〜100/月)
GCP: e2-standard-2 + Memorystore ($70〜90/月) ← 約15%安い

【本番環境（Cloud Run活用）】
GCP: Cloud Run + Memorystore
  → リクエストベース課金で低トラフィック時は大幅節約
  → 月額$1〜50程度（トラフィック依存）
```

---

## 4. Redis（キャッシュ）比較

### 4.1 マネージドRedis価格比較

| 構成 | AWS ElastiCache | GCP Memorystore | 備考 |
|------|-----------------|-----------------|------|
| **1GB Basic** | $0.017/時 (~$12.24/月) | $0.016/時 (~$11.52/月) | 同等 |
| **4GB Basic** | $0.068/時 (~$48.96/月) | $0.108/時 (~$77.76/月) | **AWS安い** |
| **8GB Standard（HA）** | $0.136/時 (~$97.92/月) | $0.216/時 (~$155.52/月) | **AWS安い** |

> **結論**: Redis単体ではAWS ElastiCacheの方が**約30〜40%安価**な場合が多い

### 4.2 機能比較

| 機能 | AWS ElastiCache | GCP Memorystore |
|------|-----------------|-----------------|
| Redis バージョン | 7.x対応 | 7.2対応 |
| クラスタモード | ◎ 完全対応 | ◎ Redis Cluster対応 |
| 自動フェイルオーバー | ◎ | ◎ |
| バックアップ | ◎ ネイティブ対応 | ○ RDBエクスポート |
| VPC統合 | ◎ | ◎ Private Service Connect |
| ベクトル検索（ANN） | △ | ○ プレビュー対応 |

### 4.3 RAGシステムでの推奨

```
【セッション管理・キャッシュ用途】
→ どちらでも可（コスト重視ならAWS）

【Qdrantと併用】
→ Redisはセッション/キャッシュに特化
→ ベクトル検索はQdrantに任せる構成が最適

【コスト最適化】
AWS: cache.t4g.micro ($9.50/月) - 開発用
GCP: 1GB Basic Tier ($11.52/月) - 開発用
```

---

## 5. コンテナ/Docker対応比較

### 5.1 サービス比較

| 項目 | AWS | GCP |
|------|-----|-----|
| **サーバーレスコンテナ** | App Runner / Fargate | **Cloud Run** ◎ |
| **Kubernetes** | EKS | GKE |
| **コンテナレジストリ** | ECR | Artifact Registry |
| **docker-compose対応** | ECS + Compose | Cloud Run (部分的) |

### 5.2 Cloud Run の優位性

Cloud RunはRAGシステムに特に適している理由：

```
【Cloud Run の特徴】
┌─────────────────────────────────────────────────────────────────┐
│  ✅ スケールtoゼロ     : トラフィックがない時は課金なし          │
│  ✅ 自動スケーリング   : リクエスト数に応じて自動調整            │
│  ✅ 100msec単位課金   : 細かい粒度で無駄なし                    │
│  ✅ Docker対応        : 既存のDockerイメージをそのままデプロイ  │
│  ✅ Gemini統合       : 同一GCPプロジェクト内でシームレス連携    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Cloud Run 価格例

```python
# Cloud Run 無料枠（月間）
FREE_TIER = {
    "cpu_seconds": 180_000,      # 50時間分
    "memory_gib_seconds": 360_000,  # 100時間分（1GiBの場合）
    "requests": 2_000_000,       # 200万リクエスト
}

# 価格（Tier 1リージョン）
PRICING = {
    "cpu_per_second": 0.000024,     # $0.086/時間
    "memory_per_gib_second": 0.0000025,  # $0.009/時間
    "request_per_million": 0.40,
}

# 例: 1日100リクエスト、各5秒処理、1vCPU/1GiB
monthly_requests = 100 * 30  # 3,000
monthly_cpu_seconds = 3000 * 5  # 15,000秒
monthly_memory_seconds = 3000 * 5  # 15,000秒

# → 無料枠内！月額 $0
```

### 5.4 docker-compose からの移行

```yaml
# 現在の docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8501:8501"
  redis:
    image: redis:7
  qdrant:
    image: qdrant/qdrant
```

```bash
# GCPへの移行手順

# 1. Cloud Run にアプリをデプロイ
gcloud run deploy rag-app \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated

# 2. Memorystore (Redis) を作成
gcloud redis instances create rag-redis \
  --size=1 \
  --region=asia-northeast1

# 3. Qdrant は GCE にデプロイ（またはCloud Run）
gcloud compute instances create-with-container qdrant-server \
  --container-image=qdrant/qdrant \
  --machine-type=e2-medium
```

---

## 6. AI/ML API連携比較

### 6.1 Gemini API 連携

| 項目 | AWS から利用 | GCP から利用 |
|------|-------------|-------------|
| **アクセス方法** | 外部API呼び出し | ネイティブ統合 |
| **認証** | APIキー管理が必要 | IAM統合（自動） |
| **ネットワーク** | パブリックインターネット経由 | VPC内部通信可能 |
| **レイテンシ** | やや高い | 低い |
| **請求** | 別請求（Googleに支払い） | GCP請求に統合 |

### 6.2 Gemini API 価格（2025年）

| モデル | 入力（100万トークン） | 出力（100万トークン） |
|--------|---------------------|---------------------|
| Gemini 2.5 Flash | $0.30 | $1.25 |
| Gemini 2.5 Pro | $1.25 | $5.00 |
| Gemini 2.0 Flash | $0.10 | $0.40 |

> **無料枠**: Gemini Developer APIは15リクエスト/分、150万トークン/日が無料

### 6.3 GCPでのGemini統合メリット

```python
# GCP内でのGemini利用（IAM認証、追加設定不要）
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# プロジェクト初期化（Cloud Run内では自動）
aiplatform.init(project="your-project", location="asia-northeast1")

# モデル利用
model = GenerativeModel("gemini-2.0-flash")
response = model.generate_content("質問")

# メリット:
# - APIキー管理不要
# - IAMで権限管理
# - VPC Service Controls対応
# - 請求統合
```

### 6.4 RAGシステムでの推奨構成

```
【GCP推奨構成】
┌─────────────────────────────────────────────────────────────────┐
│  Cloud Run (Streamlit + Python)                                 │
│       │                                                         │
│       ├──── Vertex AI (Gemini API) ← 同一VPC、低レイテンシ      │
│       │                                                         │
│       ├──── Memorystore (Redis) ← Private Service Connect      │
│       │                                                         │
│       └──── GCE (Qdrant) ← 同一VPC                             │
└─────────────────────────────────────────────────────────────────┘

【AWSハイブリッド構成】
┌─────────────────────────────────────────────────────────────────┐
│  EC2 (Streamlit + Python)                                       │
│       │                                                         │
│       ├──── Gemini API (外部) ← インターネット経由              │
│       │                                                         │
│       ├──── ElastiCache (Redis) ← VPC内                        │
│       │                                                         │
│       └──── EC2 (Qdrant) ← VPC内                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 総合コスト比較

### 7.1 月額コスト試算（小〜中規模RAGシステム）

| コンポーネント | AWS構成 | AWS費用 | GCP構成 | GCP費用 |
|---------------|---------|---------|---------|---------|
| **コンピュート** | t4g.medium | $28 | e2-medium | $24 |
| **Redis** | cache.t4g.micro | $10 | Memorystore 1GB | $12 |
| **Qdrant** | t4g.small | $15 | e2-small | $12 |
| **ストレージ** | EBS 50GB | $5 | PD 50GB | $4 |
| **ネットワーク** | - | $5 | - | $5 |
| **合計** | - | **$63/月** | - | **$57/月** |

> 差額: GCPが約**10%安い**（月額$6の節約）

### 7.2 Cloud Run活用時の試算

| シナリオ | 従来VM構成 | Cloud Run構成 | 削減率 |
|----------|-----------|--------------|--------|
| **低トラフィック**（100リクエスト/日） | $57/月 | $15/月 | **-74%** |
| **中トラフィック**（1,000リクエスト/日） | $57/月 | $30/月 | **-47%** |
| **高トラフィック**（10,000リクエスト/日） | $57/月 | $60/月 | +5% |

> **結論**: 低〜中トラフィックではCloud Runが圧倒的に有利

### 7.3 年間コスト比較

| 構成 | 年間コスト | 備考 |
|------|-----------|------|
| AWS EC2ベース | $756 | 従来構成 |
| GCP VMベース | $684 | **-$72 (-10%)** |
| GCP Cloud Run（低トラフィック） | $180 | **-$576 (-76%)** |
| GCP Cloud Run（中トラフィック） | $360 | **-$396 (-52%)** |

---

## 8. 使いやすさ・開発体験比較

### 8.1 開発者体験

| 項目 | AWS | GCP | 評価 |
|------|-----|-----|------|
| **コンソールUI** | 機能豊富だが複雑 | シンプルで直感的 | GCP ◎ |
| **CLI** | aws-cli（高機能） | gcloud（統合的） | 同等 |
| **ドキュメント** | 膨大で詳細 | 簡潔で読みやすい | 好み次第 |
| **チュートリアル** | 豊富 | Colab連携で実行可能 | GCP ◎ |
| **料金計算** | 複雑 | 比較的シンプル | GCP ◎ |

### 8.2 PyCharm連携

| 機能 | AWS | GCP |
|------|-----|-----|
| **公式プラグイン** | AWS Toolkit | Cloud Code |
| **デプロイ支援** | ◎ | ◎ |
| **デバッグ** | ○ | ◎（Cloud Runリモートデバッグ） |
| **Gemini Code Assist** | - | ◎ AIコード補完 |

### 8.3 Mac M2開発環境との親和性

```
【AWS】
- aws-cli: brew install awscli ✅
- Docker Desktop: ARM64対応 ✅
- LocalStack: M2対応 ✅

【GCP】
- gcloud: brew install google-cloud-sdk ✅
- Docker Desktop: ARM64対応 ✅
- Cloud Run エミュレータ: M2対応 ✅
- Gemini Code Assist: PyCharm統合 ✅
```

### 8.4 学習コスト

| 項目 | AWS | GCP |
|------|-----|-----|
| **サービス数** | 240+ | 150+ |
| **認定資格** | 12種類 | 11種類 |
| **初心者向け** | △ 選択肢が多すぎる | ◎ シンプル |
| **日本語リソース** | ◎ 豊富 | ○ 増加中 |

---

## 9. 移行シナリオ別推奨

### シナリオA: 完全移行（AWS → GCP）

```
【推奨度】★★★★☆

【メリット】
- Gemini APIとのネイティブ連携
- Cloud Runによるコスト最適化
- 統合された請求管理

【デメリット】
- 移行作業のコスト・リスク
- 新しいサービスの学習

【移行工数】2〜4週間

【手順】
1. GCPプロジェクト作成
2. VPC/ネットワーク設定
3. Memorystore (Redis) 構築
4. Qdrant用GCE構築
5. アプリケーションのCloud Runデプロイ
6. DNSの切り替え
7. AWSリソースの停止
```

### シナリオB: ハイブリッド構成（AWS継続 + Gemini API）

```
【推奨度】★★★★★

【メリット】
- 移行リスクなし
- 既存インフラを活用
- Gemini APIの恩恵を即座に享受

【デメリット】
- 複数クラウドの管理
- ネットワークレイテンシ（軽微）

【工数】1〜2日

【実装】
# 既存EC2からGemini APIを呼び出すだけ
pip install google-generativeai

import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-2.0-flash')
```

### シナリオC: 段階的移行

```
【推奨度】★★★★☆

【フェーズ1】（1週間）
- Gemini API統合をAWS上で実装
- 動作確認

【フェーズ2】（2週間）
- 新機能開発時にCloud Runを試用
- 並行運用で比較

【フェーズ3】（4週間）
- 本番環境をGCPに移行
- AWS環境を段階的に縮小
```

---

## 10. 結論と推奨

### 10.1 総合評価

| 評価軸 | AWS | GCP | 重要度 |
|--------|-----|-----|--------|
| コスト（VM） | 3 | 4 | 高 |
| コスト（Redis） | 4 | 3 | 中 |
| Gemini連携 | 2 | 5 | **最高** |
| サーバーレス | 3 | 5 | 高 |
| エコシステム | 5 | 4 | 中 |
| 学習コスト | 3 | 4 | 中 |
| **総合** | **3.3** | **4.2** | - |

### 10.2 最終推奨

```
┌─────────────────────────────────────────────────────────────────┐
│                       【最終推奨】                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🎯 短期（今すぐ）: シナリオB（ハイブリッド）                   │
│     → AWSを維持しつつ、Gemini APIを統合                         │
│     → リスクゼロで AI機能を強化                                 │
│                                                                  │
│  🎯 中期（3〜6ヶ月）: シナリオC（段階的移行）                   │
│     → 新機能開発でCloud Runを試用                               │
│     → コスト比較・パフォーマンス評価                            │
│                                                                  │
│  🎯 長期（6ヶ月〜）: シナリオA（完全移行）検討                  │
│     → 評価結果に基づき判断                                      │
│     → Gemini中心のアーキテクチャなら GCP一択                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 即座に実行すべきアクション

```bash
# 1. Gemini API キーの取得（無料）
# https://ai.google.dev/ にアクセス

# 2. 既存RAGシステムにGemini APIを統合
pip install google-generativeai

# 3. テスト実行
python -c "
import google.generativeai as genai
genai.configure(api_key='YOUR_KEY')
model = genai.GenerativeModel('gemini-2.0-flash')
print(model.generate_content('Hello').text)
"

# 4. GCP無料トライアル登録（$300クレジット）
# https://cloud.google.com/free
```

---

## 付録

### A. 価格計算ツール

- **AWS**: https://calculator.aws/
- **GCP**: https://cloud.google.com/products/calculator

### B. 参考リンク

- [GCP vs AWS 公式比較](https://cloud.google.com/docs/compare/aws)
- [Cloud Run 料金](https://cloud.google.com/run/pricing)
- [Gemini API](https://ai.google.dev/)
- [Memorystore 料金](https://cloud.google.com/memorystore/docs/redis/pricing)

### C. 日本リージョン情報

| プロバイダー | リージョン | ゾーン数 |
|-------------|-----------|---------|
| AWS | ap-northeast-1（東京） | 4 |
| AWS | ap-northeast-3（大阪） | 3 |
| GCP | asia-northeast1（東京） | 3 |
| GCP | asia-northeast2（大阪） | 3 |

---

**Document Version:** 1.0
**Last Updated:** 2025/12/05
**Author:** Claude (Anthropic)
