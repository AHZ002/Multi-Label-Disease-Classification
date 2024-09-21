# Multi-Label Retinal Disease Classification System

## Project Overview
This project focuses on the **multi-label classification** of retinal diseases using the **MuRed** dataset, which contains 20 different disease labels. The goal is to develop a robust system that can classify multiple retinal diseases from a single retinal image.

## Methodology
- **Architecture**: The classification system leverages a **transformer-based architecture** along with a **Multi-Scale Fusion Module (MSFM)** to capture detailed features across varying image scales.
- **Label Embeddings**: The model incorporates **BioBERT** to generate high-quality label embeddings, improving the classification accuracy across multiple diseases.
  
## Dataset
- **MuRed Dataset**: Consists of retinal images annotated with 20 different disease labels, allowing for comprehensive multi-label classification.

## Performance
- The model achieved a **ML AUC of 93.7%**, demonstrating strong performance across the 20 disease categories.
