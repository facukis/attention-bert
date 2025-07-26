# 🔍 BERT Attention Visualizer

A Python application that visualizes BERT's self-attention mechanisms using masked language modeling. This tool helps understand how BERT processes text by generating attention heatmaps for each layer and attention head.

## 🌟 Features

- **BERT Masked Language Modeling**: Predict masked tokens using pre-trained BERT
- **Attention Visualization**: Generate heatmap diagrams for all attention layers and heads
- **Interactive Text Input**: Process any text with mask tokens
- **Multiple Predictions**: Get top K predictions for masked tokens
- **Visual Attention Maps**: Color-coded attention scores between tokens

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with Python 3.11)
- pip package manager
- At least 4GB RAM recommended for BERT model

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/facukis/bert-attention-visualizer.git
   cd bert-attention-visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python mask.py
   ```

## 📁 Project Structure

```
bert-attention-visualizer/
├── mask.py                 # Main application
├── requirements.txt        # Python dependencies
├── analysis.md            # Attention analysis results
├── setup.sh               # Installation script
├── attention_diagrams/    # Generated attention visualizations
├── assets/
│   └── fonts/
│       └── OpenSans-Regular.ttf  # Font for visualizations
└── README.md              # This file
```

## 🎯 Usage

### Basic Example

1. **Run the application**:
   ```bash
   python mask.py
   ```

2. **Enter text with a mask token**:
   ```
   Text: The cat sat on the [MASK].
   ```

3. **View predictions**:
   ```
   The cat sat on the floor.
   The cat sat on the mat.
   The cat sat on the bed.
   ```

4. **Check generated attention diagrams**:
   - Files will be saved in the `attention_diagrams/` folder
   - Each diagram is named as `Attention_Layer{X}_Head{Y}.png`
   - Each diagram shows attention weights between tokens

### Advanced Examples

**Sentence completion**:
```
Text: I love to eat [MASK] for breakfast.
```

**Context understanding**:
```
Text: The [MASK] was very happy after winning the game.
```

**Technical text**:
```
Text: Machine learning is a subset of [MASK] intelligence.
```

## 🧠 How It Works

### Model Architecture
- **Base Model**: BERT-base-uncased (110M parameters)
- **Input Processing**: Tokenization with WordPiece
- **Attention Mechanism**: Multi-head self-attention (12 layers, 12 heads each)
- **Output**: Masked token predictions + attention weights

### Attention Visualization
1. **Token Processing**: Input text is tokenized into WordPiece tokens
2. **Forward Pass**: BERT processes tokens through all layers
3. **Attention Extraction**: Self-attention weights are captured
4. **Visualization**: Attention scores are converted to grayscale heatmaps
5. **Output**: PNG diagrams showing token-to-token attention

### Color Mapping
- **White/Light Gray**: High attention (important relationships)
- **Dark Gray/Black**: Low attention (less important relationships)
- **Grid Layout**: Rows and columns represent tokens

## 📊 Understanding Attention Patterns

### Common Patterns
- **Diagonal Attention**: Tokens attending to themselves
- **Syntactic Attention**: Verbs attending to subjects/objects
- **Semantic Attention**: Related concepts attending to each other
- **Positional Attention**: Adjacent tokens showing strong connections

### Layer Behaviors
- **Early Layers**: Focus on syntax and local patterns
- **Middle Layers**: Capture semantic relationships
- **Late Layers**: Task-specific attention patterns

## 🔧 Configuration

### Model Parameters
```python
MODEL = "bert-base-uncased"  # Pre-trained model
K = 3                        # Number of predictions
GRID_SIZE = 40              # Attention diagram cell size
PIXELS_PER_WORD = 200       # Space for token labels
```

### Customization Options
- Change `K` to get more/fewer predictions
- Modify `GRID_SIZE` for larger/smaller attention cells
- Adjust `PIXELS_PER_WORD` for longer token labels
- Switch to different BERT models (e.g., "bert-large-uncased")

## 📈 Dependencies

### Core Libraries
- **transformers**: Hugging Face BERT implementation
- **tensorflow**: Deep learning framework
- **PIL (Pillow)**: Image processing for visualizations
- **numpy**: Numerical computations

### Full Requirements
```txt
transformers>=4.21.0
tensorflow>=2.8.0
tf-keras>=2.15.0
Pillow>=9.0.0
numpy>=1.21.0
```

## 🎨 Output Examples

### Attention Diagrams
Each layer and head produces a diagram like:
```
    [CLS] The cat sat on the [MASK] [SEP]
[CLS]  ■    □   □   □   □   □    □     ■
The    □    ■   ■   □   □   □    □     □
cat    □    ■   ■   □   □   □    □     □
sat    □    □   □   ■   ■   ■    □     □
on     □    □   □   ■   ■   ■    □     □
the    □    □   □   □   □   ■    ■     □
[MASK] □    ■   ■   ■   ■   ■    ■     □
[SEP]  ■    □   □   □   □   □    □     ■
```

### Prediction Examples
```
Input: "The weather is [MASK] today."
Outputs:
- The weather is nice today.
- The weather is good today.
- The weather is great today.
```

## 🛠️ Development

### Adding New Features

**Custom Models**:
```python
MODEL = "bert-large-uncased"  # Use larger model
MODEL = "distilbert-base-uncased"  # Use smaller model
```

**Multiple Masks**:
Currently supports single mask token. Can be extended for multiple masks.

**Interactive Mode**:
Could add GUI interface using tkinter or web interface with Flask.

## 🐛 Troubleshooting

### Common Issues

**Memory Error**:
```
OOM when allocating tensor
```
**Solution**: Use smaller model like DistilBERT or increase system RAM

**Font Error**:
```
OSError: cannot open resource
```
**Solution**: Ensure `assets/fonts/OpenSans-Regular.ttf` exists

**No Mask Token**:
```
Input must include mask token [MASK]
```
**Solution**: Include exactly one `[MASK]` token in your input

### Performance Tips

- Use shorter sentences for faster processing
- BERT works best with 1-2 sentences at a time
- GPU acceleration speeds up processing significantly
- Cache model loading for multiple runs

## 📚 Educational Use

### Understanding BERT
This tool is perfect for:
- **NLP Students**: Visualizing attention mechanisms
- **Researchers**: Analyzing model behavior
- **Educators**: Teaching transformer architecture
- **Developers**: Debugging BERT applications

### Learning Outcomes
- Understand self-attention in transformers
- Visualize how BERT processes language
- Explore layer-wise attention patterns
- Analyze token relationships

## 🚀 Deployment

### Local Development
```bash
python mask.py
```

### Batch Processing
```python
# Process multiple sentences
sentences = ["The [MASK] is blue.", "I love [MASK]."]
for sentence in sentences:
    # Process each sentence
    pass
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add support for multiple mask tokens
- Create web interface
- Add more visualization options
- Support for different transformer models
- Batch processing capabilities

## 🎓 Academic References

- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Vaswani et al. (2017): "Attention Is All You Need"
- Clark et al. (2019): "What Does BERT Look At? An Analysis of BERT's Attention"

---

**Explore the inner workings of BERT! 🔍🤖**
