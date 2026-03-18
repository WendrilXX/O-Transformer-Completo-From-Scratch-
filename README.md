# Laboratorio 04: O Transformer Completo "From Scratch"

## Visao Geral

Este e o laboratorio final de uma serie de 4 laboratorios que exploram a arquitetura Transformer do zero. Aqui, integramos todos os componentes dos Labs 01-03 em uma **arquitetura Encoder-Decoder completa e funcional** para realizar traducao autoregressiva fim-a-fim.

### Labs na Serie

1. **Lab 01** - Mecanismo de Self-Attention
   - Implementacao do Scaled Dot-Product Attention
   - Calculo de Q, K, V e pesos de atencao

2. **Lab 02** - Transformer Encoder
   - Pilha de 6 camadas de Encoder
   - Add & Norm (Residual + Layer Normalization)
   - Position-wise Feed-Forward Network

3. **Lab 03** - Transformer Decoder
   - Mascara Causal (Look-Ahead Masking)
   - Cross-Attention (Ponte Encoder-Decoder)
   - Loop Auto-regressivo de Inferencia

4. **Lab 04** - Transformer Completo <- **Voce esta aqui**
   - Integracao Encoder-Decoder
   - Inferencia fim-a-fim com loop auto-regressivo
   - Teste com toy sequence ("Thinking Machines")

---

## Objetivos de Aprendizagem

1. [OK] Aplicar engenharia de software para integrar modulos separados em uma topologia coerente
2. [OK] Garantir fluxo correto de tensores passando por todas as camadas
3. [OK] Acoplar loop auto-regressivo de inferencia na saida do Decoder

---

## Estrutura do Projeto

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
├── README.md                        # Esta documentacao
└── .git/                            # Repositorio Git
```

---

## Arquitetura do Transformer

```
INPUT (Toy Sequence)
    |
    v
    +------------------+
    |  ENCODER         |
    |  (6 Camadas)     |
    |  Self-Attention  |
    |  Add & Norm      |
    |  FFN             |
    |  (repetido 6x)   |
    +------------------+
          |
          v Z
          |
    +-----+-----+
    |           |
    v           v
DECODER    ENCODER Z
(6 Cam)    (referencia)
|
Masked Self-Attn
Add & Norm
Cross-Attn <-- integra Z
Add & Norm
FFN
Add & Norm
|
v
Linear -> vocab
Softmax
|
v
P(next_token)
```

---

## Componentes Implementados

### 1. Refatoracao & Integracao (Tarefa 1)

#### `scaled_dot_product_attention(Q, K, V, mask=None)`
- Implementacao generica do mecanismo de atencao
- Suporta mascara causal opcional
- Usa softmax numericamente estavel

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

#### `MultiHeadAttention`
- Multiplas cabecas de atencao em paralelo
- Concatenacao e projecao final
- Usada tanto em Self-Attention quanto Cross-Attention

#### `RedeFFN`
- Position-wise Feed-Forward Network
- Expansao: d_model -> 4*d_model -> d_model
- Ativacao ReLU no meio

#### `layer_norm(tensor, epsilon)`
- Layer Normalization (normaliza ultima dimensao)
- Evita divergencias durante treinamento

### 2. Montando a Pilha do Encoder (Tarefa 2)

#### `EncoderBlock`
Fluxo dentro do bloco:
- Input X
- Self-Attention (Q, K, V all from X)
- Add & Norm (X + Self-Attn(X))
- FFN
- Add & Norm (prev + FFN(prev))
- Output

#### `TransformerEncoder`
- Stack de `N=6` camadas de `EncoderBlock`
- Input: tensor com Positional Encoding ja somado
- Output: representacao contextualizada Z

### 3. Montando a Pilha do Decoder (Tarefa 3)

#### `DecoderBlock`
Fluxo dentro do bloco:
- Input Y
- MASKED Self-Attention (Q, K, V all from Y, WITH causal mask)
- Add & Norm (Y + Masked-Self-Attn(Y))
- Cross-Attention (Q from Y, K,V from Z_encoder, NO mask)
- Add & Norm
- FFN
- Add & Norm
- Output

#### `TransformerDecoder`
- Stack de `N=6` camadas de `DecoderBlock`
- Recebe tanto a sequencia alvo Y quanto Z do Encoder
- Aplica mascara causal para evitar "cheating" (ver o futuro)

### 4. A Prova Final (Tarefa 4)

#### `TransformerCompleto`
Integra Encoder e Decoder:
- Z = encoder.forward(encoder_input)
- decoder_out = decoder.forward(decoder_input, Z)
- logits = linear_projection(decoder_out)
- return logits

#### Loop Auto-regressivo
- sequence = [<START>]
- while len(sequence) < max_length:
-   probs = generate_next_token_probs(sequence, Z_encoder)
-   token_id = argmax(probs)
-   token = vocab.get_token(token_id)
-   sequence.append(token)
-   if token == <EOS>: break
- return sequence

---

## Como Usar

### 1. Clonar ou navegar para o repositorio

```bash
cd c:\Users\Usuario\Desktop\ELetiva2\O-Transformer-Completo-From-Scratch-
```

### 2. Executar o teste completo

```bash
python lab04_transformer_completo.py
```

**Saida esperada:**
- Forward pass do Encoder (2 camadas de teste)
- Forward pass do Decoder (2 camadas de teste)
- Loop auto-regressivo de 20 passos
- Sequencia gerada com probabilidades

### 3. Importar o modulo em outro script

```python
from lab04_transformer_completo import (
    TransformerCompleto,
    MockVocabulary,
    create_toy_embeddings,
    PositionalEncoding
)

vocab = MockVocabulary(vocab_size=5000)
modelo = TransformerCompleto(d_model=64, vocab_size=5000)
embeddings = create_toy_embeddings(seq_len=2, d_model=64)
encoder_input = PositionalEncoding.add_positional_encoding(embeddings, 64)
Z = modelo.encoder.forward(encoder_input)
```

---

## Parametros Configuravel

| Parametro | Valor Padrao | Descricao |
|-----------|--------------|-----------|
| `d_model` | 64 (teste) / 512 (paper) | Dimensao dos embeddings |
| `vocab_size` | 5000+ | Tamanho do vocabulario |
| `num_camadas` | 6 | Numero de camadas no Encoder e Decoder |
| `num_heads` | 8 | Numero de cabecas de atencao |
| `d_ff` | 4 * d_model | Dimensao interna da FFN |
| `max_seq_length` | 20 | Limite maximo de geracao |

---

## Complexidade Computacional

### Forward Pass Encoder
```
Tempo: O(N * L^2 * d_model^2)
Espaco: O(N * L * d_model)

Onde:
  N = numero de camadas
  L = comprimento da sequencia
  d_model = dimensao dos embeddings
```

### Forward Pass Decoder
```
Tempo: O(N * M^2 * d_model^2) + O(N * M * L * d_model)
Espaco: O(N * (M + L) * d_model)

Onde:
  L = comprimento encoder
  M = comprimento decoder
```

### Loop Auto-regressivo
```
Tempo: O(gen_length * (complexidade_decoder))
Espaco: O(gen_length * d_model)
```

---

## Validacoes Incluidas

O script `test_transformer_inference()` valida:

[OK] Shape Consistency
- Input -> Encoder: (1, 2, 64)
- Encoder -> Z: (1, 2, 64)
- Z + Decoder Input -> Decoder: (1, seq_len, 64)
- Decoder Output -> Logits: (1, seq_len, vocab_size)

[OK] Fluxo de Tensores
- Encoder sem mascara (ve contexto completo)
- Decoder com mascara causal (ve apenas passado)
- Cross-Attention integra Z corretamente

[OK] Inferencia Auto-regressiva
- Gera sequencia token-por-token
- Para em <EOS> ou max_length
- Probabilidades somam 1.0

[OK] Valores Numericos
- Softmax produz distribuicoes validas
- Sem NaN ou infinitos
- Valores gradualmente evoluem no loop

---

## Exemplo de Saida Esperada

```
================================================================================
LABORATORIO 04: TRANSFORMER COMPLETO - PROVA FINAL
================================================================================

Configuracoes:
  d_model: 64
  vocab_size: 5000
  num_camadas: 2
  num_heads: 4
  max_seq_length: 20

PASSO 1: ENCODER INPUT (Toy Sequence: 'Thinking Machines')
Frase de entrada: 'Thinking Machines'

PASSO 2: FORWARD PASS DO ENCODER
Encoder Camada 1: Z.shape = (1, 2, 64)
Encoder Camada 2: Z.shape = (1, 2, 64)

PASSO 3: LOOP AUTO-REGRESSIVO DE INFERENCIA
Passo 1: word_937 (p=0.002628)
Passo 2: word_636 (p=0.002682)
...
Passo 20: word_4218 (p=0.003520)

OK TESTE COMPLETADO COM SUCESSO
================================================================================
```

---

## Integracao com Labs Anteriores

### Lab 01 -> Lab 04
```python
def scaled_dot_product_attention(query, key, value):
    scores = query @ key.T / sqrt(d_k)
    return softmax(scores) @ value
```

### Lab 02 -> Lab 04
```python
class EncoderBlock:
    def forward(self, X):
        Z = self.add_and_norm(X, self.atencao(X, X, X))
        return self.add_and_norm(Z, self.ffn(Z))
```

### Lab 03 -> Lab 04
```python
create_causal_mask()          # Usada no DecoderBlock
generate_next_token_probs()   # Loop auto-regressivo
```

---

## Versionamento Git

```bash
git tag v1.0
git push origin v1.0
```

O commit avaliado deve possuir a tag `v1.0` com timestamp de conclusao do lab.

---

## Conceitos-Chave Aplicados

| Conceitos | Referencia | Implementacao |
|-----------|-----------|----------------|
| Scaled Dot-Product Attention | Vaswani et al. (2017) | `scaled_dot_product_attention()` |
| Multi-Head Attention | Vaswani et al. (2017) | `MultiHeadAttention` |
| Positional Encoding | Vaswani et al. (2017) | `PositionalEncoding.add_positional_encoding()` |
| Layer Normalization | Ba et al. (2016) | `layer_norm()` |
| Residual Connections | He et al. (2015) | `add_and_norm()` |
| Masked Self-Attention | Vaswani et al. (2017) | `create_causal_mask()` em DecoderBlock |
| Autoregressive Generation | Graves (2013) | `generate_next_token_probs()` + loop |

---

## Troubleshooting

### Erro: "NaN ou Infinito nos valores"
- Verificar estabilidade numerica em `softmax_estavel()`
- Aumentar `epsilon` em `layer_norm()`

### Erro: "Shapes nao conferem"
- Verificar dimensoes esperadas em docstrings
- Usar `print(tensor.shape)` entre layers

### Inferencia muito lenta
- Reduzir `d_model` ou `num_camadas` para teste
- Usar NumPy compilado (ex: Intel MKL)

### Nao para em <EOS>
- Aumentar `max_seq_length`
- Verificar se `vocab.EOS` esta no vocabulario

---

## Nota sobre Inteligencia Artificial

Partes geradas/complementadas com IA, revisadas por Wendril Gabriel

- [OK] Codigo verificado para correcao matematica
- [OK] Logica de Encoder-Decoder compreendida e validada
- [OK] Fluxo de tensores testado em todas as camadas
- [OK] Mascara causal e Cross-Attention implementadas corretamente
- [OK] Loop auto-regressivo de inferencia funcional
- [OK] Documentacao tecnica revisada e completa

Todos os conceitos foram verificados e o modelo foi testado com sucesso.

---

## Referencias

1. "Attention is All You Need" - Vaswani et al., 2017
   - Paper original do Transformer
   - Define arquitetura Encoder-Decoder
   - Link: https://arxiv.org/abs/1706.03762

2. Causal Masking - OpenAI GPT Papers
   - Tecnica para impedir acesso ao futuro
   - Essencial para geracao autoregressiva

3. Layer Normalization - Ba et al., 2016
   - Alternativa ao Batch Norm para RNNs/Transformers
   - Link: https://arxiv.org/abs/1607.06450

---

## Checklist de Entrega

- [x] Codigo implementado em Python/NumPy
- [x] Encoder com 6 camadas (teste com 2)
- [x] Decoder com Masked Self-Attention
- [x] Cross-Attention Encoder-Decoder
- [x] Loop Autoregressivo funcional
- [x] Teste com toy sequence
- [x] README.md documentado
- [x] Git commit com tag v1.0
- [x] Nota sobre IA incluida

---

## Autor

Laboratorio de Processamento de Linguagem Natural (NLP)
Disciplina: Deep Learning - Transformers from Scratch
Data: Marco 2026

---

Status: CONCLUIDO E TESTADO
