FinDoc-Extractor

FinDoc-Extractor is a full-stack web application designed to automate financial document processing. Users can upload invoices, receipts, or other financial documents (PDFs/Images), and the system will use Optical Character Recognition (OCR) and a custom-trained Natural Language Processing (NLP) model to extract key information.

The extracted data is presented on a clean dashboard, enabling users to quickly view and manage important financial details and export them for further use

✨ Key Features

*   **Secure User Authentication**: Secure sign-up and login system using JWT.
*   **Drag-and-Drop File Upload**: Modern interface for uploading PDF and image files.
*   **Intelligent Data Extraction**: A powerful backend pipeline that:
    1.  Converts documents to text using **OCR (Tesseract)**.
    2.  Identifies key entities using a **custom-trained NER model (Hugging Face Transformers)**.
*   **Key Entities Extracted**:
    *   Company Name
    *   Invoice Number
    *   Total Amount
    *   Due Date
    *   Account Number
*   **Interactive Dashboard**: View, track, and manage all uploaded documents and their extracted metadata in a structured table.
*   **RESTful API**: A secure API endpoint for programmatic access to the extraction service.
*   **Data Export**: Export extracted data to CSV for easy integration with other tools.
*   
🛠️ Tech Stack

*   **Backend**: **Python** with **FastAPI**
*   **Machine Learning**: **Hugging Face Transformers** (for custom NER model), **PyTesseract** (for OCR)
*   **Frontend**: **React** with **Material-UI** for a professional component library
*   **Database**: **PostgreSQL**
*   **Deployment**: **Azure App Service** (Backend), **Azure Static Web Apps** (Frontend), **Azure Blob Storage** (File Storage)
*   **CI/CD**: **GitHub Actions** for automated build and deployment.
