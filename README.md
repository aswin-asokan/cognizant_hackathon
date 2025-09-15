# ShipGuard: Shipment Fraud Identification
ShipGuard is a project designed to detect fraudulent shipments using machine learning models. The system uses a React-based web interface for tracking shipments and providing fraud alerts. The backend is built with Python and Firebase for data storage, processing, and API services.

## Features

- Fraud Detection: Utilizes multiple AI models, including Isolation Forest, LSTM Autoencoder, Random Forest, and GRU, to identify fraudulent patterns
- Web Interface: A React web interface provides a dashboard for uploading shipment data and viewing AI-powered fraud detection results
- Data Handling: Python libraries are integrated with Firebase for data storage and processing. Firestore is used to store user data, shipment records, and AI detection results
- User Authentication: Firebase Admin SDK is used for secure user authentication and access to storage

## Technology Stack
### Frontend

**Core Technologies:**

- React v18.3.1: Component-based UI library.
- TypeScript: A type-safe JavaScript superset.
-Vite: A next-generation build tool.

### Backend

**Core Technologies:**

- Node.js + Express.js: Used to handle REST API requests and authentication.
- Firebase Admin SDK: Provides secure user authentication and access to storage.
- Firestore: Used to store data such as users, shipment records, and AI detection results.
- Python & Firebase: Chosen for real-time database updates, built-in user authentication, and easy integration with AI/ML models.

## AI Models & Datasets
The project faced the challenge of not having a dataset with explicit fraud labels. To overcome this, the team relied on unsupervised methods and created synthetic datasets.

- Isolation Forest: Used for unsupervised anomaly detection and to generate a synthetic fraud dataset from the LaDe-d dataset.
- LSTM Autoencoder: Used for anomaly detection on the logistics dataset, flagging anomalies that show higher reconstruction error.
- Random Forest: Trained on a synthetic dataset to identify fraudulent shipments.
- GRU (Gated Recurrent Unit): Captures sequential data and temporal patterns to detect abnormal event sequences like unusual delays or detours.

## Limitations & Future Scope
- Limitations: The system may produce false positives or negatives and has limited explainability, making it hard to understand why a shipment was flagged.
- Future Scope: Future improvements could include using hybrid models (Isolation Forest + GRU/LSTM) for better accuracy, integrating with IoT for real-time tracking, and adding alerts and dashboards for managers.

## References

For more information on the datasets used, you can check out the following:

- LaDe-d Dataset:    
https://huggingface.co/datasets/Cainiao-AI/LaDe-D 

- Logistics Risk and Fraud:     
https://www.kaggle.com/datasets/helgrind113/logistics-risk-and-fraud 
