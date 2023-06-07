import matplotlib.pyplot as plt
import pickle
import gzip

with gzip.open('./result/trained_threshold.pkl', 'rb') as f:
    trained_data = pickle.load(f)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(trained_data[i].reshape(28, 28), cmap='gray')
    plt.title(f'{i}')
    plt.axis('off')

plt.tight_layout()
plt.show()