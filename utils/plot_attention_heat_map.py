import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_attention_heatmap(attn_weights, head_index, layer_id):
    
    seq_len = attn_weights.shape[-1]    

    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    attn_weights *= (1 - mask)

    head_attn = attn_weights[0, head_index]

    cmap = plt.cm.viridis
    cmap.set_under(color='white')

    plt.figure(figsize=(10, 8))
    plt.imshow(head_attn, cmap=cmap, aspect='auto', vmin=1e-10) 
    plt.colorbar(label='Attention Score')
    
    plt.title(f'Attention Heatmap for Head {head_index}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Sequence')

    # for i in range(head_attn.shape[0]):
    #     for j in range(head_attn.shape[1]):
    #         value = head_attn[i, j]
    #         text_color = 'white' if value > 0.5 else 'black'
    #         plt.text(j, i, f'{value:.2e}', ha='center', va='center', color=text_color, fontsize=6)

    pdf_name = "Attention_weights_layer_" + str(layer_id) + "_head_" + str(head_index) + ".pdf"

    plt.savefig(pdf_name, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.show()

if __name__ == '__main__':

    seq_len = 30

    attn_weights = np.random.rand(1, 2, seq_len, seq_len)

    # mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    # attn_weights *= (1 - mask)

    plot_attention_heatmap(attn_weights, head_index=0, layer_id=0)



