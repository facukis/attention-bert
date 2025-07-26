import sys
import os
import tensorflow as tf 

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

# Output directory for attention diagrams
OUTPUT_DIR = "attention_diagrams"


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Attention diagrams will be saved to: {OUTPUT_DIR}/")
    
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    print("🤖 Loading BERT model...")
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    print("🔄 Processing text and generating attention...")
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    print("\n🎯 Top predictions:")
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for i, token in enumerate(top_tokens, 1):
        prediction = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
        print(f"{i}. {prediction}")

    # Visualize attentions
    print(f"\n🖼️  Generating attention diagrams...")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    diagram_count = visualize_attentions(tokens, result.attentions)
    print(f"✅ Generated {diagram_count} attention diagrams in {OUTPUT_DIR}/")
    print(f"📊 Layers: {len(result.attentions)}, Heads per layer: {result.attentions[0].shape[1]}")


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    # TODO: Implement this function
    mask_token_index = tf.where(inputs.input_ids == mask_token_id)
    if mask_token_index.shape[0] == 0:
        return None
    return mask_token_index[0][1].numpy()




def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # FIX: Invert the logic to make higher attention scores lighter.
    # Use tf.round() for TensorFlow compatibility.
    if attention_score < 0 or attention_score > 1:
        raise ValueError("Attention score must be in the range [0, 1].")
    
    # Round using TensorFlow's round function to handle tensors
    gray_value = int(tf.round(attention_score * 255).numpy())  # .numpy() converts to a NumPy value

    return (gray_value, gray_value, gray_value)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    diagram_count = 0
    for layer in range(len(attentions)):
        for head in range(attentions[layer].shape[1]):
            generate_diagram(
                layer + 1,
                head + 1,
                tokens,
                attentions[layer][0][head]
            )
            diagram_count += 1
    return diagram_count



def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    filename = f"Attention_Layer{layer_number}_Head{head_number}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    print(f"  💾 Saved: {filename}")



if __name__ == "__main__":
    main()