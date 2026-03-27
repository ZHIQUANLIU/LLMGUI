# Shakespeare Poem LLM Training App

## Project Overview
- **Project name**: Shakespeare Poem Trainer
- **Type**: Python CLI Application
- **Core functionality**: Train a character-level language model on William Shakespeare's poems and generate new poems in Shakespearean tone
- **Target users**: Developers learning about LLM training

## Functionality Specification

### Core Features
1. **Data Collection**: Fetch Shakespeare poems from Project Gutenberg
2. **Character-level Tokenization**: Convert text to character indices
3. **RNN Model**: Simple character-level RNN with embedding and linear layers
4. **Training**: Train the model with detailed epoch-by-epoch logging
5. **Generation**: Generate new poems with step-by-step thinking logs

### User Interactions
1. Run training script to train the model
2. Run generation script to create new poems
3. View detailed logs of both processes

### Data Handling
- Fetch poems from Project Gutenberg (Shakespeare's sonnets and poems)
- Store training data locally
- Save trained model checkpoints

### Edge Cases
- Handle network errors when fetching data
- Handle empty or malformed responses
- Handle CUDA availability (fallback to CPU)

## Technical Architecture

### Model Architecture
- Character embedding layer (embed_dim=256)
- GRU/LSTM layer (hidden_dim=512, num_layers=2)
- Dropout for regularization
- Linear output layer

### Training Configuration
- Sequence length: 100 characters
- Batch size: 64
- Learning rate: 0.001
- Epochs: 50 (configurable)
- Optimizer: Adam
- Loss: CrossEntropyLoss

### Logging Details
- Training: Every epoch - loss, perplexity, sample output
- Generation: Every character prediction - probabilities, top choices, selected character

## Acceptance Criteria
1. Successfully fetch and parse Shakespeare poems
2. Train model with decreasing loss over epochs
3. Generate coherent poems in Shakespearean style
4. Print all training steps with detailed logs
5. Print generation thinking process with probabilities
