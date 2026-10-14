# 🚀 CinemaBox 70B Cloud Server
### Powered by 5 TB Google Storage & Google Colab GPU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/princesharmaofficial1-hub/cinemabox-colab/blob/main/deploy_70b_google_storage.ipynb)

Turn your **5 TB Google Storage** and a free **Google Colab GPU** into a dedicated **OpenAI-compatible 70B LLM API**.

---

## ⚡ 1-Click Launch

1. Click the **[Open in Colab](https://colab.research.google.com/github/princesharmaofficial1-hub/cinemabox-colab/blob/main/deploy_70b_google_storage.ipynb)** badge above.
2. In Google Colab:
   - Make sure runtime is GPU (**Runtime** > **Change runtime type** > **T4 GPU**).
   - Click **Runtime** > **Run all** (or press `Ctrl + F9`).
   - When asked to connect Google Drive, click **Connect to Google Drive**.
3. Copy the live **Cloudflare URL** printed at the bottom of the notebook!

---

## 📱 Use in CinemaBox Flutter App

1. Open CinemaBox on your Android device.
2. Go to **Settings** > **On-Device Intelligence**.
3. Select the **AirLLM 70B (Cloud)** tab.
4. Set:
   - **Host API Endpoint:** `https://<YOUR-CLOUDFLARE-URL>.trycloudflare.com/v1/chat/completions`
   - **Bearer API Key:** `sk-cinemabox70b`
5. Tap **Test Connection & Generate** to verify!

---

## 💻 Use in VS Code (Roo Code / Continue / Cline)

Add this provider configuration to your VS Code AI extension:

```json
{
  "title": "CinemaBox 70B (Colab)",
  "provider": "openai",
  "model": "Llama-3.3-70B-Instruct",
  "apiBase": "https://<YOUR-CLOUDFLARE-URL>.trycloudflare.com/v1",
  "apiKey": "sk-cinemabox70b"
}
```
