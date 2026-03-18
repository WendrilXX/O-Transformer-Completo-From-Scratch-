 # -*- coding: utf-8 -*-
"""
LABORATÓRIO 04: O TRANSFORMER COMPLETO "FROM SCRATCH"
=====================================================

Objetivo: Integrar todos os componentes dos Labs 01-03 em uma arquitetura
Encoder-Decoder funcional para realizar tradução autoregressiva completa.

Estrutura:
1. Refatoração & Integração (imports dos labs anteriores)
2. Montando a Pilha do Encoder
3. Montando a Pilha do Decoder
4. A Prova Final (Inferência Fim-a-Fim)

Autor: Laboratório de Processamento de Linguagem Natural
Data: 2026
"""

import numpy as np
import math
from typing import List, Tuple, Dict

np.random.seed(2026)  # Reprodutibilidade

# ============================================================================
# TAREFA 1: REFATORAÇÃO E INTEGRAÇÃO (OS BLOCOS DE MONTAR)
# ============================================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implementa o mecanismo de Scaled Dot-Product Attention.
    
    Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Parameters
    ----------
    query : np.ndarray
        Matriz de Queries (Q) de forma (batch, seq_len, d_k)
    key : np.ndarray
        Matriz de Keys (K) de forma (batch, seq_len, d_k)
    value : np.ndarray
        Matriz de Values (V) de forma (batch, seq_len, d_v)
    mask : np.ndarray, optional
        Máscara causal ou padding de forma (seq_len, seq_len)
    
    Returns
    -------
    np.ndarray
        Saída da atenção de forma (batch, seq_len, d_v)
    np.ndarray
        Pesos de atenção para visualização
    """
    # Passo 1: Calcular similaridade QK^T
    scores = np.matmul(query, key.transpose(0, 2, 1))
    
    # Passo 2: Escalar pela raiz da dimensão da chave
    d_k = key.shape[-1]
    scaling_factor = np.sqrt(d_k)
    scaled_scores = scores / scaling_factor
    
    # Passo 3: Aplicar máscara (se fornecida)
    if mask is not None:
        scaled_scores = scaled_scores + mask
    
    # Passo 4: Aplicar softmax
    attention_weights = softmax_estavel(scaled_scores, axis=-1)
    
    # Passo 5: Ponderar valores
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


def softmax_estavel(x, axis=-1):
    """
    Softmax numericamente estável (evita overflow/underflow).
    
    Trata -inf como 0 naturalmente via exp(-inf) = 0.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # -inf se torna 0 após exponencial
    exp_x[np.isinf(x) & (x < 0)] = 0
    
    soma = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / soma


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria uma máscara causal para impedir acesso ao futuro.
    
    Args:
        seq_len (int): Comprimento da sequência
    
    Returns:
        np.ndarray: Matriz [seq_len, seq_len] com -inf na diagonal superior
    
    Exemplo (seq_len=3):
        [[ 0. -inf -inf]
         [ 0.  0. -inf]
         [ 0.  0.  0. ]]
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices_from(mask, k=1)] = -np.inf
    return mask


def layer_norm(tensor, epsilon=1e-6):
    """
    Layer Normalization: normaliza última dimensão (features).
    
    Formula: LayerNorm(x) = (x - μ) / √(σ² + ε)
    
    Args:
        tensor : np.ndarray de qualquer shape
        epsilon : float para evitar divisão por zero
    
    Returns:
        np.ndarray : tensor normalizado, mesmo shape
    """
    media = np.mean(tensor, axis=-1, keepdims=True)
    variancia = np.var(tensor, axis=-1, keepdims=True)
    normalizado = (tensor - media) / np.sqrt(variancia + epsilon)
    return normalizado


def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)


class RedeFFN:
    """
    Position-wise Feed-Forward Network.
    
    Arquitetura:
    - Linear: d_model → d_ff (expansão)
    - ReLU
    - Linear: d_ff → d_model (contração)
    
    Típico: d_ff = 4 * d_model
    """
    
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else 4 * d_model
        
        # Primeira camada (expansão)
        self.W1 = np.random.randn(d_model, self.d_ff) * 0.1
        self.b1 = np.zeros(self.d_ff)
        
        # Segunda camada (contração)
        self.W2 = np.random.randn(self.d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, X):
        """
        Args:
            X : (batch, seq_len, d_model)
        Returns:
            output : (batch, seq_len, d_model)
        """
        # Camada 1 + ReLU
        hiddenstate = X @ self.W1 + self.b1
        hiddenstate_ativado = relu(hiddenstate)
        
        # Camada 2
        output = hiddenstate_ativado @ self.W2 + self.b2
        
        return output


class MultiHeadAttention:
    """
    Multi-Head Self-Attention ou Cross-Attention.
    
    Implementa: h cabeças de atenção em paralelo
    Output: concatenação + projeção linear
    """
    
    def __init__(self, d_model, num_heads=8):
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projeções lineares para Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        # Projeção de saída
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V : (batch, seq_len, d_model)
            mask : (seq_len, seq_len) opcional
        
        Returns:
            output : (batch, seq_len, d_model)
            attention_weights : para visualização
        """
        batch_size = Q.shape[0]
        
        # Projetar Q, K, V
        Q_proj = Q @ self.W_q  # (batch, seq_len, d_model)
        K_proj = K @ self.W_k
        V_proj = V @ self.W_v
        
        # Reshape para múltiplas cabeças
        Q_heads = Q_proj.reshape(batch_size, -1, self.num_heads, self.d_k)
        K_heads = K_proj.reshape(batch_size, -1, self.num_heads, self.d_k)
        V_heads = V_proj.reshape(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpor para (batch, num_heads, seq_len, d_k)
        Q_heads = Q_heads.transpose(0, 2, 1, 3)
        K_heads = K_heads.transpose(0, 2, 1, 3)
        V_heads = V_heads.transpose(0, 2, 1, 3)
        
        # Aplicar attention em cada cabeça
        output_heads = []
        for h in range(self.num_heads):
            Q_h = Q_heads[:, h, :, :]  # (batch, seq_len, d_k)
            K_h = K_heads[:, h, :, :]
            V_h = V_heads[:, h, :, :]
            
            # Scaled dot-product attention
            attn_out, _ = scaled_dot_product_attention(Q_h, K_h, V_h, mask)
            output_heads.append(attn_out)
        
        # Concatenar cabeças: (batch, seq_len, d_model)
        output_concat = np.concatenate(output_heads, axis=-1)
        
        # Projeção final
        output = output_concat @ self.W_o
        
        return output


# ============================================================================
# TAREFA 2: MONTANDO A PILHA DO ENCODER
# ============================================================================

class EncoderBlock:
    """
    Um bloco (camada) do Encoder Transformer.
    
    Fluxo:
    1. input X
    2. Self-Attention
    3. Add & Norm
    4. FFN
    5. Add & Norm
    6. output
    """
    
    def __init__(self, d_model, num_heads=8, d_ff=None):
        self.d_model = d_model
        
        # Componentes
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = RedeFFN(d_model, d_ff)
    
    def add_and_norm(self, X, sublayer_output):
        """
        Conexão residual + Layer Normalization
        Output = LayerNorm(X + Sublayer(X))
        """
        residual = X + sublayer_output
        normalizado = layer_norm(residual)
        return normalizado
    
    def forward(self, X, mask=None):
        """
        Args:
            X : (batch, seq_len, d_model) - tensor de entrada
            mask : None para Encoder (usa self-attention sem máscara)
        
        Returns:
            output : (batch, seq_len, d_model)
        """
        # Step 1: Self-Attention (Q, K, V all from X)
        attn_output = self.self_attn.forward(X, X, X, mask=None)
        X_after_attn = self.add_and_norm(X, attn_output)
        
        # Step 2: FFN
        ffn_output = self.ffn.forward(X_after_attn)
        X_final = self.add_and_norm(X_after_attn, ffn_output)
        
        return X_final


class TransformerEncoder:
    """
    Encoder Transformer completo: N camadas de EncoderBlock empilhadas.
    
    O paper original usa N=6.
    """
    
    def __init__(self, d_model, num_camadas=6, num_heads=8, d_ff=None):
        self.d_model = d_model
        self.num_camadas = num_camadas
        
        # Stack de camadas
        self.camadas = [
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_camadas)
        ]
    
    def forward(self, X, verbose=False):
        """
        Args:
            X : (batch, seq_len, d_model) com positional encoding já adicionado
        
        Returns:
            Z : (batch, seq_len, d_model) - representação contextualizada
        """
        Z = X.copy()
        
        for idx, camada in enumerate(self.camadas, start=1):
            if verbose:
                print(f"  Encoder Camada {idx}: Z.shape = {Z.shape}")
            
            Z = camada.forward(Z, mask=None)
        
        if verbose:
            print(f"  Encoder Final: Z.shape = {Z.shape}")
        
        return Z


# ============================================================================
# TAREFA 3: MONTANDO A PILHA DO DECODER
# ============================================================================

class DecoderBlock:
    """
    Um bloco (camada) do Decoder Transformer.
    
    Fluxo:
    1. input Y
    2. MASKED Self-Attention (causal mask)
    3. Add & Norm
    4. Cross-Attention (Q from Y, K,V from Z encoder)
    5. Add & Norm
    6. FFN
    7. Add & Norm
    8. output
    """
    
    def __init__(self, d_model, num_heads=8, d_ff=None):
        self.d_model = d_model
        
        # Componentes
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = RedeFFN(d_model, d_ff)
    
    def add_and_norm(self, X, sublayer_output):
        """
        Conexão residual + Layer Normalization
        Output = LayerNorm(X + Sublayer(X))
        """
        residual = X + sublayer_output
        normalizado = layer_norm(residual)
        return normalizado
    
    def forward(self, Y, Z_encoder, causal_mask):
        """
        Args:
            Y : (batch, seq_len_decoder, d_model) - entrada do decoder
            Z_encoder : (batch, seq_len_encoder, d_model) - saída do encoder
            causal_mask : (seq_len_decoder, seq_len_decoder) - máscara causal
        
        Returns:
            output : (batch, seq_len_decoder, d_model)
        """
        # Step 1: MASKED Self-Attention (Y → Y com máscara causal)
        masked_attn_output = self.masked_self_attn.forward(
            Y, Y, Y, mask=causal_mask
        )
        Y_after_masked_attn = self.add_and_norm(Y, masked_attn_output)
        
        # Step 2: Cross-Attention (Q from Y, K,V from Z)
        cross_attn_output = self.cross_attn.forward(
            Y_after_masked_attn, Z_encoder, Z_encoder, mask=None
        )
        Y_after_cross_attn = self.add_and_norm(Y_after_masked_attn, cross_attn_output)
        
        # Step 3: FFN
        ffn_output = self.ffn.forward(Y_after_cross_attn)
        Y_final = self.add_and_norm(Y_after_cross_attn, ffn_output)
        
        return Y_final


class TransformerDecoder:
    """
    Decoder Transformer completo: N camadas de DecoderBlock empilhadas.
    
    Mantém referência ao Z do Encoder para Cross-Attention.
    """
    
    def __init__(self, d_model, num_camadas=6, num_heads=8, d_ff=None):
        self.d_model = d_model
        self.num_camadas = num_camadas
        
        # Stack de camadas
        self.camadas = [
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_camadas)
        ]
    
    def forward(self, Y, Z_encoder, verbose=False):
        """
        Args:
            Y : (batch, seq_len_decoder, d_model) com positional encoding
            Z_encoder : (batch, seq_len_encoder, d_model) saída do encoder
        
        Returns:
            output : (batch, seq_len_decoder, d_model)
        """
        seq_len_decoder = Y.shape[1]
        causal_mask = create_causal_mask(seq_len_decoder)
        
        Y_current = Y.copy()
        
        for idx, camada in enumerate(self.camadas, start=1):
            if verbose:
                print(f"  Decoder Camada {idx}: Y.shape = {Y_current.shape}")
            
            Y_current = camada.forward(Y_current, Z_encoder, causal_mask)
        
        if verbose:
            print(f"  Decoder Final: Y.shape = {Y_current.shape}")
        
        return Y_current


# ============================================================================
# TAREFA 4: A PROVA FINAL (INFERÊNCIA)
# ============================================================================

class MockVocabulary:
    """
    Vocabulário fictício para teste.
    """
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        
        self.START = "<START>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        self.PAD = "<PAD>"
        
        # Tokens especiais + palavras fictícias
        self.tokens = (
            [self.START, self.EOS, self.UNK, self.PAD] +
            [f"word_{i}" for i in range(4, vocab_size)]
        )
        
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
    
    def get_id(self, token):
        return self.token2id.get(token, self.token2id[self.UNK])
    
    def get_token(self, token_id):
        return self.id2token.get(token_id, self.UNK)


class PositionalEncoding:
    """
    Positional Encoding (simplificado para teste).
    Em prod: usar PE = sin/cos do paper original.
    Aqui: apenas valores aleatórios para manter shape.
    """
    
    @staticmethod
    def add_positional_encoding(X, d_model):
        """
        Args:
            X : (batch, seq_len, d_model)
        
        Returns:
            X_with_pe : (batch, seq_len, d_model)
        """
        batch_size, seq_len, dim = X.shape
        
        # Para simplificar no teste: adicionar pequenos valores aleatórios
        # Em um modelo real: usar seno/cosseno conforme Vaswani et al.
        pe = np.random.randn(batch_size, seq_len, d_model) * 0.01
        
        return X + pe


class TransformerCompleto:
    """
    Transformer Encoder-Decoder Completo.
    
    Fluxo:
    1. Encoder: input_ids → positional_encoding → encoder_layers → Z
    2. Decoder: target_ids + Z → positional_encoding → decoder_layers → logits
    3. Projeta últim token de saída para vocab_size
    4. Softmax → probabilidades
    """
    
    def __init__(self, d_model, vocab_size, num_camadas=6, num_heads=8, d_ff=None):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Encoder e Decoder
        self.encoder = TransformerEncoder(d_model, num_camadas, num_heads, d_ff)
        self.decoder = TransformerDecoder(d_model, num_camadas, num_heads, d_ff)
        
        # Camada de output: d_model → vocab_size
        self.W_output = np.random.randn(d_model, vocab_size) * 0.1
        self.b_output = np.zeros(vocab_size)
    
    def forward(self, encoder_input, decoder_input, verbose=False):
        """
        Forward pass completo.
        
        Args:
            encoder_input : (batch, seq_encoder, d_model) com PE já somado
            decoder_input : (batch, seq_decoder, d_model) com PE já somado
            verbose : bool para imprime shapes
        
        Returns:
            logits : (batch, seq_decoder, vocab_size)
            Z : (batch, seq_encoder, d_model)
        """
        if verbose:
            print("\n" + "="*70)
            print("FORWARD PASS - TRANSFORMER COMPLETO")
            print("="*70)
        
        # === ENCODER ===
        if verbose:
            print("\n[ENCODER]")
        Z = self.encoder.forward(encoder_input, verbose=verbose)
        
        # === DECODER ===
        if verbose:
            print("\n[DECODER]")
        decoder_output = self.decoder.forward(decoder_input, Z, verbose=verbose)
        
        # === OUTPUT PROJECTION ===
        if verbose:
            print("\n[OUTPUT PROJECTION]")
        batch_size, seq_len, d_model = decoder_output.shape
        
        # Pegar todos os tokens (não apenas o último, para poder usar em loop)
        logits = decoder_output @ self.W_output + self.b_output
        # shape: (batch, seq_len, vocab_size)
        
        if verbose:
            print(f"  Logits shape: {logits.shape}")
        
        return logits, Z
    
    def generate_next_token_probs(self, current_sequence_embeddings, Z_encoder):
        """
        Gera distribuição de probabilidades para o próximo token.
        
        Args:
            current_sequence_embeddings : (batch, seq_len_atual, d_model)
            Z_encoder : (batch, seq_encoder, d_model)
        
        Returns:
            probs : (vocab_size,) - distribuição normalizada
        """
        # Forward pass
        logits, _ = self.forward(Z_encoder, current_sequence_embeddings, verbose=False)
        
        # Pegar logits do último token apenas
        logits_last_token = logits[0, -1, :]  # (vocab_size,)
        
        # Softmax
        probs = softmax_estavel(logits_last_token.reshape(1, -1), axis=-1)[0]
        
        return probs


def create_toy_embeddings(sequence_length, d_model):
    """
    Cria embeddings fictícios para teste (sem usar embeddings table real).
    
    Args:
        sequence_length : int
        d_model : int
    
    Returns:
        embeddings : (1, sequence_length, d_model)
    """
    batch_size = 1
    embeddings = np.random.randn(batch_size, sequence_length, d_model) * 0.1
    return embeddings


def test_transformer_inference():
    """
    PROVA FINAL: Teste de inferência fim-a-fim.
    
    Tarefa: Simular tradução da sequência "Thinking Machines"
            para uma sequência alvo fictícia usando inferência auto-regressiva.
    """
    
    print("\n" + "="*80)
    print("LABORATÓRIO 04: TRANSFORMER COMPLETO - PROVA FINAL")
    print("="*80)
    
    # === Configuração ===
    d_model = 64  # Reduzido para teste (paper usa 512)
    vocab_size = 5000
    num_camadas = 2  # Reduzido para teste (paper usa 6)
    num_heads = 4  # Reduzido
    max_seq_length = 20
    
    print(f"\nConfigurações:")
    print(f"  d_model: {d_model}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  num_camadas: {num_camadas}")
    print(f"  num_heads: {num_heads}")
    print(f"  max_seq_length: {max_seq_length}")
    
    # === Criar vocabulário e modelo ===
    vocab = MockVocabulary(vocab_size)
    modelo = TransformerCompleto(
        d_model=d_model,
        vocab_size=vocab_size,
        num_camadas=num_camadas,
        num_heads=num_heads
    )
    
    # === ENCODER INPUT ===
    print(f"\n{'='*80}")
    print("PASSO 1: ENCODER INPUT (Toy Sequence: 'Thinking Machines')")
    print(f"{'='*80}")
    
    # Simular sequência "Thinking Machines" com 2 tokens
    seq_encoder = 2
    encoder_input_embeddings = create_toy_embeddings(seq_encoder, d_model)
    encoder_input = PositionalEncoding.add_positional_encoding(
        encoder_input_embeddings, d_model
    )
    
    print(f"Frase de entrada: 'Thinking Machines'")
    print(f"  Número de tokens: {seq_encoder}")
    print(f"  Embeddings shape: {encoder_input.shape}")
    print(f"  Primeiro token embedding (primeiros 5 dims): {encoder_input[0, 0, :5]}")
    print(f"  Segundo token embedding (primeiros 5 dims): {encoder_input[0, 1, :5]}")
    
    # === FORWARD PASS DO ENCODER ===
    print(f"\n{'='*80}")
    print("PASSO 2: FORWARD PASS DO ENCODER")
    print(f"{'='*80}\n")
    
    Z = modelo.encoder.forward(encoder_input, verbose=True)
    
    print(f"\nSaída do Encoder (Z):")
    print(f"  Shape: {Z.shape}")
    print(f"  Primeiros valores do token 1: {Z[0, 0, :5]}")
    print(f"  Primeiros valores do token 2: {Z[0, 1, :5]}")
    print(f"  → Z contém a representação contextualizada da entrada")
    
    # === LOOP AUTO-REGRESSIVO DO DECODER ===
    print(f"\n{'='*80}")
    print("PASSO 3: LOOP AUTO-REGRESSIVO DE INFERÊNCIA")
    print(f"{'='*80}\n")
    
    current_sequence = [vocab.START]
    generated_sequence = []
    
    print(f"Algoritmo: Argmax (Greedy Decoding)")
    print(f"Critérios de parada:")
    print(f"  1. Gerar token <EOS>")
    print(f"  2. Atingir max_seq_length ({max_seq_length})\n")
    
    print(f"{'Passo':<8} {'Token Gerado':<20} {'Probabilidade':<15} {'Sequência':<40}")
    print("-" * 80)
    
    # Loop auto-regressivo
    for step in range(max_seq_length):
        # Criar embeddings da sequência atual
        seq_len_decoder = len(current_sequence)
        decoder_input_embeddings = create_toy_embeddings(seq_len_decoder, d_model)
        decoder_input = PositionalEncoding.add_positional_encoding(
            decoder_input_embeddings, d_model
        )
        
        # Gerar probabilidades para próximo token
        probs = modelo.generate_next_token_probs(decoder_input, Z)
        
        # Argmax: selecionar token com maior probabilidade
        token_id = np.argmax(probs)
        token = vocab.get_token(token_id)
        prob_value = probs[token_id]
        
        # Adicionar à sequência
        current_sequence.append(token)
        generated_sequence.append((token, prob_value))
        
        # Mostrar progresso
        seq_display = " ".join(current_sequence[-4:])
        if len(current_sequence) > 4:
            seq_display = "... " + seq_display
        
        print(f"{step+1:<8} {token:<20} {prob_value:<15.6f} {seq_display:<40}")
        
        # Critério de parada
        if token == vocab.EOS:
            print(f"\n✓ Token <EOS> gerado no passo {step+1}")
            break
    else:
        print(f"\n⚠ Atingiu comprimento máximo ({max_seq_length} tokens)")
    
    # === RESULTADO FINAL ===
    print(f"\n{'='*80}")
    print("RESULTADO FINAL")
    print(f"{'='*80}")
    
    print(f"\nSequência de entrada: Thinking Machines")
    print(f"Sequência gerada ({len(generated_sequence)} tokens):")
    print(f"  {' '.join([t[0] for t in generated_sequence])}")
    
    tokens_probs = [(t[0], f"{t[1]:.4f}") for t in generated_sequence]
    print(f"\nTokens com probabilidades:")
    for i, (token, prob) in enumerate(tokens_probs[:10], 1):
        print(f"  {i}. {token:<25} (p={prob})")
    
    if len(tokens_probs) > 10:
        print(f"  ... ({len(tokens_probs) - 10} mais tokens)")
    
    print(f"\n✓ TESTE COMPLETADO COM SUCESSO")
    print(f"{'='*80}\n")
    
    return modelo, vocab


if __name__ == "__main__":
    modelo, vocab = test_transformer_inference()
