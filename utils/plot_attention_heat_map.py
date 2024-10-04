import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_attention_heatmap(attn_weights, head_index, layer_id):
    # Select the attention weights for the specified head
    head_attn = attn_weights[0, head_index]

    # Define a colormap with white for zero values
    cmap = plt.cm.viridis
    cmap.set_under(color='white')

    # Create a heatmap using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(head_attn, cmap=cmap, aspect='auto', vmin=1e-10)  # Set vmin to a small value to use 'under' color
    # plt.matshow(head_attn, cmap=cmap, fignum=1, vmin=1e-10, edgecolors='black', linewidths=0.5)
    plt.colorbar(label='Attention Score')
    
    plt.title(f'Attention Heatmap for Head {head_index}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Sequence')

    # Add text annotations
    for i in range(head_attn.shape[0]):
        for j in range(head_attn.shape[1]):
            value = head_attn[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            plt.text(j, i, f'{value:.2e}', ha='center', va='center', color=text_color, fontsize=6)


    pdf_name = "Attention_weights_layer_" + str(layer_id) + "_head_" + str(head_index) + ".pdf"

    plt.savefig(pdf_name, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.show()

if __name__ == '__main__':
    attn_weights = np.random.rand(1, 2, 20, 20)

    mask = np.triu(np.ones((20, 20)), k=1)
    attn_weights *= (1 - mask)

    plot_attention_heatmap(attn_weights, head_index=0)



