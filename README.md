# 🚀 CinemaBox 141B MoE & 72B Multi-Modal Agentic Server
### Powered by 5 TB Google Storage & Google Colab GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/princesharmaofficial1-hub/cinemabox-colab/blob/main/deploy_70b_google_storage.ipynb)

Turn your **5 TB Google Storage** and a **Colab GPU** into an always-accessible **OpenAI-compatible Multi-Modal Agentic API** (`/v1/chat/completions`) and interactive **Web Agent Studio** (`/`).

---

## 🧠 Heavyweight & Multi-Modal Model Catalog
All models are downloaded and stored permanently in your 5 TB Google Drive (`/content/drive/MyDrive/CinemaBox_Models/`):

| Model | Parameters | Type | Context | Best For |
|---|---|---|---|---|
| **Mixtral 8x22B Instruct** | 141B Total (39B Active) | Mixture of Experts (MoE) | 65k | Heavyweight 120B+ class reasoning & complex problem solving |
| **Qwen 2.5 72B Instruct** | 72.7B | Dense Flagship | 128k | #1 ranked open-weight reasoning model worldwide |
| **Qwen 2 VL 72B** | 72B | Multi-Modal (Vision + Text) | 32k | Movie scenes, posters, UI designs, code diagrams, screenshots |
| **Qwen 2.5 Coder 32B** | 32.5B | Code Agent | 128k | Claude Code & Roo Code software engineering workflows |
| **LLaMA 3.3 70B Instruct** | 70B | Dense Flagship | 128k | Meta's latest state-of-the-art general intelligence |

---

## 🤖 Full End-to-End Claude Code Protocol
- **Visible Sub-Agents:** Transparent orchestration between **Architect**, **Vision Analyzer**, **Coder**, and **Critic/Reviewer** agents.
- **Visible Step Traces:** Real-time visibility into `<thought>`, tool invocations, inputs, and outputs.
- **Interactive Web Studio (`/`):** Drag-and-drop vision upload, live agent flowchart, and conversation tree.
- **Universal API (`/v1/chat/completions`):** Direct drop-in for **CinemaBox App** and **VS Code (Roo Code / Continue / Cline)**.

---

## ⚡ 1-Click Launch
1. Click the **[Open in Colab](https://colab.research.google.com/github/princesharmaofficial1-hub/cinemabox-colab/blob/main/deploy_70b_google_storage.ipynb)** badge.
2. In Google Colab:
   - Ensure Runtime type is GPU (**Runtime** > **Change runtime type** > **T4 GPU**).
   - In Step 3, choose your model (Mixtral 141B, Qwen 72B, or Qwen2-VL Multi-Modal).
   - Press **`Ctrl + F9`** (**Runtime** > **Run all**).
   - Click **Connect to Google Drive** when prompted.
3. Your live URLs will be printed in Step 4:
   - **🎨 Web Agent Studio:** `https://xxxx.trycloudflare.com`
   - **📱 CinemaBox App:** `https://xxxx.trycloudflare.com/v1/chat/completions`
   - **💻 VS Code Base:** `https://xxxx.trycloudflare.com/v1`
