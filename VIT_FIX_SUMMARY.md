# Vision Transformer Loss Function Fix

## Problema Identificado

O modelo ViT apresentava:
- **Loss negativo explosivo**: -402K → -39 bilhões através dos epochs
- **Acurácia estagnada**: ~25% (praticamente aleatório)
- **Recall perfeito**: 1.0 em todas as épocas
- **Precision constante**: ~75%
- **Padrão**: Modelo sempre prevendo a mesma classe

## Causa Raiz

A instabilidade numérica foi causada pela combinação de:

1. **Sigmoid na camada de saída**: `Dense(1, activation="sigmoid")`
2. **Loss como string**: `loss='binary_crossentropy'`

Esta combinação causa:
- Cálculo duplo de sigmoid (uma vez na camada, outra na loss)
- Instabilidade numérica quando valores se aproximam de 0 ou 1
- Overflow/underflow em cálculos logarítmicos

## Solução Aplicada

### Modificação 1: Remover Sigmoid da Camada de Saída

**ANTES:**
```python
outputs = layers.Dense(1, activation="sigmoid")(features)
```

**DEPOIS:**
```python
outputs = layers.Dense(1)(features)  # No activation - returns logits
```

### Modificação 2: Usar `from_logits=True` na Loss

**ANTES:**
```python
vit_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[...]
)
```

**DEPOIS:**
```python
vit_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[...]
)
```

## Por que `from_logits=True` é melhor?

### Implementação com `from_logits=False` (RUIM):
```
1. Camada output: sigmoid(x) → valor entre [0, 1]
2. Loss: BinaryCrossentropy calcula: -y*log(sigmoid_output) - (1-y)*log(1-sigmoid_output)
3. Problema: log(valores próximos a 0) = -∞ (underflow)
```

### Implementação com `from_logits=True` (BOM):
```
1. Camada output: retorna logits (valores sem restrição)
2. Loss: BinaryCrossentropy calcula internamente de forma numericamente estável
3. Benefício: Evita cálculo duplo de sigmoid, melhor precisão numérica
```

## Resultados Esperados

### Antes da correção:
- Loss: valores negativos crescentes
- Acurácia: ~25% (aleatório)
- Convergência: Nenhuma

### Depois da correção (esperado):
- Loss: valores positivos decrescentes (~0.1-0.5 → ~0.01-0.05)
- Acurácia: 96-99%
- Convergência: 10-20 épocas
- Recall e Precision: Balanceados e altos

## Importância para Inferência

Após treinar com `from_logits=True`, para obter **probabilidades** entre 0 e 1:

```python
# Opção 1: Aplicar sigmoid manualmente
predictions = tf.nn.sigmoid(model.predict(x))

# Opção 2: Criar um modelo de wrapper com sigmoid
output = layers.Dense(1)(previous_layer)
sigmoid_output = layers.Activation('sigmoid')(output)
inference_model = keras.Model(inputs=inputs, outputs=sigmoid_output)
```

## Arquivos Modificados

- `malaria_detection_project.ipynb`
  - Célula 29: Função `build_vit_model()` - removido sigmoid
  - Célula 31: Compilação do ViT - adicionado `from_logits=True`

## Referências

- [TensorFlow BinaryCrossentropy Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)
- [Best Practices for Neural Network Training](https://cs231n.github.io/neural-networks-3/)
