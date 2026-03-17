# Laboratório 04: O Transformer Completo "From Scratch"

## 📚 Visão Geral

Este é o laboratório final de uma série de 4 laboratórios que exploram a arquitetura Transformer do zero. Aqui, integramos todos os componentes dos Labs 01-03 em uma **arquitetura Encoder-Decoder completa e funcional** para realizar tradução autoregressiva fim-a-fim.

### Labs na Série

1. **Lab 01** - Mecanismo de Self-Attention
   - Implementação do Scaled Dot-Product Attention
   - Cálculo de Q, K, V e pesos de atenção

2. **Lab 02** - Transformer Encoder
   - Pilha de 6 camadas de Encoder
   - Add & Norm (Residual + Layer Normalization)
   - Position-wise Feed-Forward Network

3. **Lab 03** - Transformer Decoder
   - Máscara Causal (Look-Ahead Masking)
   - Cross-Attention (Ponte Encoder-Decoder)
   - Loop Auto-regressivo de Inferência

4. **Lab 04** - Transformer Completo ← **Você está aqui**
   - Integração Encoder-Decoder
   - Inferência fim-a-fim com loop auto-regressivo
   - Teste com toy sequence ("Thinking Machines")

---

## 🎯 Objetivos de Aprendizagem

1. ✅ **Aplicar engenharia de software** para integrar módulos separados em uma topologia coerente
2. ✅ **Garantir fluxo correto** de tensores passando por todas as camadas
3. ✅ **Acoplar loop auto-regressivo** de inferência na saída do Decoder

---

## 📁 Estrutura do Projeto

```
O-Transformer-Completo-From-Scratch/
├── lab01/                           # Lab 01 (importado via git)
│   ├── laboratorio_1.py
│   └── ...
├── lab02/                           # Lab 02 (importado via git)
│   ├── encoder_transformer.py
│   └── ...
├── lab03/                           # Lab 03 (importado via git)
│   ├── decoder_laboratory.py
│   └── ...
├── lab04_transformer_completo.py    # Lab 04 COMPLETO (este arquivo)
├── README.md                        # Esta documentação
└── .git/                            # Repositório Git
```

---

## 🏗️ Arquitetura do Transformer

```
INPUT (Toy Sequence)
    ↓
┌─────────────────────────────┐
│                             │
│   ENCODER (6 Camadas)       │
│                             │
│  ┌──────────────────────┐   │
│  │ Self-Attention       │   │  ← Sem máscara causal
│  │ Add & Norm           │   │
│  │ FFN                  │   │
│  │ Add & Norm           │   │
│  └──────────────────────┘   │
│  (repetido 6x)              │
│                             │
└─────────────────────────────┘
    ↓ Z (Representação codificada)
    │
    ├──────────────────────────────────┐
    ↓                                  ↓
┌─────────────────────────────┐    ┌─────────────┐
│                             │    │             │
│   DECODER (6 Camadas)       │    │  ENCODER Z  │
│   (com sequ. alvo)          │    │             │
│                             │    └─────────────┘
│  ┌──────────────────────┐   │
│  │ Masked Self-Attention│   │  ← COM máscara causal
│  │ Add & Norm           │   │
│  │ Cross-Attention      │◄──┼─── (queries do decoder)
│  │ Add & Norm           │   │
│  │ FFN                  │   │
│  │ Add & Norm           │   │
│  └──────────────────────┘   │
│  (repetido 6x)              │
│                             │
└─────────────────────────────┘
    ↓ Decoder Output
    ↓
┌─────────────────────────────┐
│ Linear (d_model → vocab)    │
│ Softmax                     │
└─────────────────────────────┘
    ↓ P(next_token | context)
    ↓
LOOP AUTO-REGRESSIVO (Inferência)
    ← Gera próximo token
    ← Adiciona à sequência
    ← Repete até <EOS> ou max_length
```

---

## 🔧 Componentes Implementados

### 1. **Refatoração & Integração (Tarefa 1)**

#### `scaled_dot_product_attention(Q, K, V, mask=None)`
- Implementação genérica do mecanismo de atenção
- Suporta máscara causal opcional
- Usa softmax numericamente estável

```python
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

#### `MultiHeadAttention`
- Múltiplas cabeças de atenção em paralelo
- Concatenação e projeção final
- Usada tanto em Self-Attention quanto Cross-Attention

#### `RedeFFN`
- Position-wise Feed-Forward Network
- Expansão: d_model → 4*d_model → d_model
- Ativação ReLU no meio

#### `layer_norm(tensor, epsilon)`
- Layer Normalization (normaliza última dimensão)
- Evita divergências durante treinamento

### 2. **Montando a Pilha do Encoder (Tarefa 2)**

#### `EncoderBlock`
Fluxo dentro do bloco:
```
Input X
  ↓
Self-Attention (Q, K, V all from X)
  ↓
Add & Norm (X + Self-Attn(X))
  ↓
FFN
  ↓
Add & Norm (prev + FFN(prev))
  ↓
Output
```

#### `TransformerEncoder`
- Stack de `N=6` camadas de `EncoderBlock`
- Input: tensor com Positional Encoding já somado
- Output: representação contextualizada Z

**Fórmula da saída:**
```
Z = Encoder_N(...Encoder_2(Encoder_1(X + PE)...))
```

### 3. **Montando a Pilha do Decoder (Tarefa 3)**

#### `DecoderBlock`
Fluxo dentro do bloco:
```
Input Y
  ↓
MASKED Self-Attention (Q, K, V all from Y, WITH causal mask)
  ↓
Add & Norm (Y + Masked-Self-Attn(Y))
  ↓
Cross-Attention (Q from Y, K,V from Z_encoder, NO mask)
  ↓
Add & Norm
  ↓
FFN
  ↓
Add & Norm
  ↓
Output
```

#### `TransformerDecoder`
- Stack de `N=6` camadas de `DecoderBlock`
- Recebe tanto a sequência alvo Y quanto Z do Encoder
- Aplica máscara causal para evitar "cheating" (ver o futuro)

### 4. **A Prova Final (Tarefa 4)**

#### `TransformerCompleto`
Integra Encoder e Decoder:
```python
class TransformerCompleto:
    def forward(encoder_input, decoder_input):
        Z = encoder.forward(encoder_input)              # Codifica entrada
        decoder_out = decoder.forward(decoder_input, Z) # Decodifica
        logits = linear_projection(decoder_out)         # → vocab_size
        return logits
```

#### `generate_next_token_probs()`
Gera distribuição de probabilidades para o próximo token:
```python
# 1. Embed sequência atual
Y_embedded = embed(current_sequence)

# 2. Forward pass completo
logits = modelo.forward(Z_encoder, Y_embedded)

# 3. Pegar logits do último token
logits_last = logits[:, -1, :]

# 4. Softmax → probabilidades
probs = softmax(logits_last)

return probs
```

#### Loop Auto-regressivo
```python
sequence = [<START>]

while len(sequence) < max_length:
    # Gerar distribuição P(next_token | sequence, encoder)
    probs = generate_next_token_probs(sequence, Z_encoder)
    
    # Argmax: token com maior probabilidade
    token_id = argmax(probs)
    token = vocab.get_token(token_id)
    
    # Adicionar à sequência
    sequence.append(token)
    
    # Parar se <EOS>
    if token == <EOS>:
        break

return sequence
```

---

## 🚀 Como Usar

### 1. Clonar ou navegar para o repositório

```bash
cd c:\Users\Usuario\Desktop\ELetiva2\O-Transformer-Completo-From-Scratch-
```

### 2. Executar o teste completo

```bash
python lab04_transformer_completo.py
```

**Saída esperada:**
- Forward pass do Encoder (2 camadas de teste)
- Forward pass do Decoder (2 camadas de teste)
- Loop auto-regressivo de 20 passos
- Sequência gerada com probabilidades

### 3. Importar o módulo em outro script

```python
from lab04_transformer_completo import (
    TransformerCompleto,
    MockVocabulary,
    create_toy_embeddings,
    PositionalEncoding
)

# Criando o modelo
vocab = MockVocabulary(vocab_size=5000)
modelo = TransformerCompleto(d_model=64, vocab_size=5000)

# Usando para inferência
embeddings = create_toy_embeddings(seq_len=2, d_model=64)
encoder_input = PositionalEncoding.add_positional_encoding(embeddings, 64)
Z = modelo.encoder.forward(encoder_input)

# Decoder iterativo
decoder_input = create_toy_embeddings(seq_len=1, d_model=64)
decoder_input = PositionalEncoding.add_positional_encoding(decoder_input, 64)
probs = modelo.generate_next_token_probs(decoder_input, Z)
next_token_id = np.argmax(probs)
```

---

## 📊 Parâmetros Configuráveis

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| `d_model` | 64 (teste) / 512 (paper) | Dimensão dos embeddings |
| `vocab_size` | 5000+ | Tamanho do vocabulário |
| `num_camadas` | 6 | Número de camadas no Encoder e Decoder |
| `num_heads` | 8 | Número de cabeças de atenção |
| `d_ff` | 4 * d_model | Dimensão interna da FFN |
| `max_seq_length` | 20 | Limite máximo de geração |

---

## 🧮 Complexidade Computacional

### Forward Pass Encoder
```
Tempo: O(N * L² * d_model²)
Espaço: O(N * L * d_model)

Onde:
  N = número de camadas
  L = comprimento da sequência
  d_model = dimensão dos embeddings
```

### Forward Pass Decoder
```
Tempo: O(N * M² * d_model²) + O(N * M * L * d_model)
Espaço: O(N * (M + L) * d_model)

Onde:
  L = comprimento encoder
  M = comprimento decoder
```

### Loop Auto-regressivo
```
Tempo: O(gen_length * (complexidade_decoder))
Espaço: O(gen_length * d_model)
```

---

## 🔍 Validações Incluídas

O script `test_transformer_inference()` valida:

✅ **Shape Consistency**
- Input → Encoder: (1, 2, 64)
- Encoder → Z: (1, 2, 64)
- Z + Decoder Input → Decoder: (1, seq_len, 64)
- Decoder Output → Logits: (1, seq_len, vocab_size)

✅ **Fluxo de Tensores**
- Encoder sem máscara (vê contexto completo)
- Decoder com máscara causal (vê apenas passado)
- Cross-Attention integra Z corretamente

✅ **Inferência Auto-regressiva**
- Gera sequência token-por-token
- Para em <EOS> ou max_length
- Probabilidades somam 1.0

✅ **Valores Numéricos**
- Softmax produz distribuições válidas
- Sem NaN ou infinitos
- Valores gradualmente evoluem no loop

---

## 📝 Exemplo de Saída Esperada

```
================================================================================
LABORATÓRIO 04: TRANSFORMER COMPLETO - PROVA FINAL
================================================================================

Configurações:
  d_model: 64
  vocab_size: 5000
  num_camadas: 2
  num_heads: 4
  max_seq_length: 20

================================================================================
PASSO 1: ENCODER INPUT (Toy Sequence: 'Thinking Machines')
================================================================================

Frase de entrada: 'Thinking Machines'
  Número de tokens: 2
  ...

================================================================================
PASSO 2: FORWARD PASS DO ENCODER
================================================================================

  Encoder Camada 1: Z.shape = (1, 2, 64)
  Encoder Camada 2: Z.shape = (1, 2, 64)
  Encoder Final: Z.shape = (1, 2, 64)
  
Saída do Encoder (Z):
  Shape: (1, 2, 64)
  ...

================================================================================
PASSO 3: LOOP AUTO-REGRESSIVO DE INFERÊNCIA
================================================================================

Passo    Token Gerado         Probabilidade   Sequência
--------
1        word_937             0.002628        <START> word_937
2        word_636             0.002682        <START> word_937 word_636
...
20       word_4218            0.003520        ... word_1991 word_4218

⚠ Atingiu comprimento máximo (20 tokens)

================================================================================
RESULTADO FINAL
================================================================================

Sequência de entrada: Thinking Machines
Sequência gerada (20 tokens):
  word_937 word_636 word_4218 ... word_4218
  
✓ TESTE COMPLETADO COM SUCESSO
================================================================================
```

---

## 🔗 Integração com Labs Anteriores

### Lab 01 → Lab 04
```python
# Código do Lab 01 (genérico)
def scaled_dot_product_attention(query, key, value):
    scores = query @ key.T / sqrt(d_k)
    return softmax(scores) @ value

# Integrado no Lab 04
class MultiHeadAttention:
    def forward(self, Q, K, V, mask=None):
        # Usa scaled_dot_product_attention internamente
        output, _ = scaled_dot_product_attention(Q, K, V, mask)
        return output @ self.W_o
```

### Lab 02 → Lab 04
```python
# Encoder do Lab 02
class BlocoEncoderLayer:
    def forward(self, X):
        Z = self.add_and_norm(X, self.atencao(X, X, X))
        return self.add_and_norm(Z, self.ffn(Z))

# Reutilizado diretamente no Lab 04 como `EncoderBlock`
```

### Lab 03 → Lab 04
```python
# Máscaras causal e Cross-attention do Lab 03
create_causal_mask()      # Usada no DecoderBlock
cross_attention()         # Implementada em MultiHeadAttention

# Loop auto-regressivo
# generate_with_argmax() adaptado em generate_next_token_probs()
```

---

## 💾 Versionamento Git

```bash
# Commit com tag v1.0
git tag v1.0
git push origin v1.0
```

O commit avaliado deve possuir a tag `v1.0` com timestamp de conclusão do lab.

---

## 🎓 Conceitos-Chave Aplicados

| Conceitos | Referência | Implementação |
|-----------|-----------|----------------|
| **Scaled Dot-Product Attention** | Vaswani et al. (2017) | `scaled_dot_product_attention()` |
| **Multi-Head Attention** | Vaswani et al. (2017) | `MultiHeadAttention` |
| **Positional Encoding** | Vaswani et al. (2017) | `PositionalEncoding.add_positional_encoding()` |
| **Layer Normalization** | Ba et al. (2016) | `layer_norm()` |
| **Residual Connections** | He et al. (2015) | `add_and_norm()` |
| **Masked Self-Attention** | Vaswani et al. (2017) | `create_causal_mask()` em DecoderBlock |
| **Autoregressive Generation** | Graves (2013) | `generate_next_token_probs()` + loop |

---

## 🐛 Troubleshooting

### Erro: "NaN ou Infinito nos valores"
- Verificar estabilidade numérica em `softmax_estavel()`
- Aumentar `epsilon` em `layer_norm()`

### Erro: "Shapes não conferem"
- Verificar dimensões esperadas em docstrings
- Usar `print(tensor.shape)` entre layers

### Inferência muito lenta
- Reduzir `d_model` ou `num_camadas` para teste
- Usar NumPy compilado (ex: Intel MKL)

### Não para em <EOS>
- Aumentar `max_seq_length`
- Verificar se `vocab.EOS` está no vocabulário

---

## 📚 Referências

1. **"Attention is All You Need"** - Vaswani et al., 2017
   - Paper original do Transformer
   - Define arquitetura Encoder-Decoder
   - Link: https://arxiv.org/abs/1706.03762

2. **Causal Masking** - OpenAI GPT Papers
   - Técnica para impedir acesso ao futuro
   - Essencial para geração autoregressiva

3. **Layer Normalization** - Ba et al., 2016
   - Alternativa ao Batch Norm para RNNs/Transformers
   - Link: https://arxiv.org/abs/1607.06450

---

## ✅ Checklist de Entrega

- [x] Código implementado em Python/NumPy
- [x] Encoder com 6 camadas (teste com 2)
- [x] Decoder com Masked Self-Attention
- [x] Cross-Attention Encoder-Decoder
- [x] Loop Autoregressivo funcional
- [x] Teste com toy sequence
- [x] README.md documentado
- [ ] Git commit com tag v1.0

---

## 👨‍💼 Autor

**Laboratório de Processamento de Linguagem Natural (NLP)**

Disciplina: Deep Learning - Transformers from Scratch

Data: Março 2026

---

**Status:** ✅ **CONCLUÍDO E TESTADO**

