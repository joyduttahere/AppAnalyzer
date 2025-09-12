# App Analyzer: AI-Powered Google Play Store Review Insights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joyduttahere/AppAnalyzer/blob/main/Share_AppAnalyzer.ipynb)

*Click the badge above to launch the interactive demo in Google Colab.*

This repository contains the source code for the App Analyzer.
---

## Overview

**App Analyzer** is a web-based intelligence dashboard that leverages state-of-the-art AI to analyze user reviews from the Google Play Store. It provides product managers, marketers, and developers with actionable insights by comparing review data across different time periods.

This tool helps you understand user sentiment, identify persisting pain points, discover newly surfaced issues, track resolved problems, and categorize constructive feature requests, all powered by advanced sentiment analysis and large language models (LLMs).

## Features

-   **Comparative Analysis:** Analyze and compare two distinct date ranges to track the evolution of user feedback over time.
-   **AI-Powered Sentiment Analysis:** Utilizes a RoBERTa-based model to accurately classify review sentiment as positive, negative, or neutral.
-   **Automated Topic Categorization:** Intelligently categorizes reviews into predefined topics like "App Stability," "User Experience," "Billing," and more.
-   **Dynamic LLM Summarization:** Employs powerful Large Language Models (like Mistral-7B and OpenChat) to generate executive summaries and detailed insights for critical issues and feature requests.
-   **Insightful Dashboard:**
    -   **Persisting Problems:** Highlights issues that continue to affect users in both time periods.
    -   **Newly Surfaced Problems:** Identifies new issues that have appeared in the recent period.
    -   **Resolved Problems:** Validates bug fixes and improvements by showing which old complaints have disappeared.
    -   **Feature Request Theming:** Groups constructive user suggestions into actionable themes.
-   **Flexible Model Selection:** Allows users to choose from different LLMs to balance analysis depth, speed, and resource usage.

## How to Run in Google Colab

This project is designed to be easily run in a free Google Colab environment.

### Prerequisites

Before you begin, you will need:
1.  A **Google Account** to use Colab.
2.  A free **[ngrok Account](https://dashboard.ngrok.com/signup)** to create a secure public URL for the web app.

### Quickstart Instructions

1.  **Click the "Open in Colab" Badge** at the top of this README.
2.  **Select a Runtime with a GPU:** The notebook requires a GPU to run the AI models. In the Colab menu, navigate to **Runtime → Change runtime type** and select **T4 GPU** as the hardware accelerator.
3.  **Add Your Secret Keys:** The notebook will prompt you to add your `NGROK_AUTHTOKEN` using the Colab Secrets Manager (🔑). This is a secure way to handle your credentials.
4.  **Run the Cells in Order:** Execute the notebook cells from top to bottom. The code will clone the repository, install dependencies, authenticate your services, and launch the web application.
5.  **Access the App:** Once the final cell is run, an `ngrok.io` URL will be displayed. Click this link to open and use the App Analyzer dashboard in a new browser tab.

## Local Development Setup

While designed for Colab, you can also run this project locally if you have a compatible environment (Python 3.10+ and an NVIDIA GPU with CUDA).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/joyduttahere/AppAnalyzer.git
    cd AppAnalyzer
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your secret keys:
    ```
    NGROK_AUTHTOKEN="your_ngrok_authtoken_here"
    ```

5.  **Run the application:**
    ```bash
    python app.py
    ```

## Technology Stack

-   **Backend:** Flask (Python)
-   **Frontend:** HTML, CSS, JavaScript
-   **AI / ML:**
    -   PyTorch
    -   Hugging Face Transformers
    -   `bitsandbytes` for model quantization
-   **Data Scraping:** `google-play-scraper`
-   **Deployment:** Google Colab, Ngrok
