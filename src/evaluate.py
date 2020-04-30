from utils import *
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from model.SARI import calculate
from config import model_config as config

sf = SmoothingFunction()

def generate(decoder, sentence, output_lang, p):
    with torch.no_grad():
        output_tensor = tensorFromSentence(output_lang, sentence)
        output_tensor = pad_sequences(output_tensor, config['MAX_LENGTH'])
        output_length = output_tensor.size()[0]
        output_tensor = output_tensor.unsqueeze(0)

        decoder_hidden = decoder.initHidden(1)
        #encoder_hidden = encoder.initHidden(1, True)
        #print(input_tensor)
        c = 0
        a = []
        for i in output_tensor[0]:
            if i.item() == 0:
                a.append(0)
            else:
                c += 1
                a.append(1)
        output_tensor_len = torch.tensor([c], device=device)
        output_tensor_mask = get_mask(output_tensor_len)
        batch_size = 1
        decoder_input = output_tensor.narrow(1, 0, 1)
        decoded_words = []
        for di in range(config['MAX_LENGTH']):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, False)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze(0).detach()
        return decoded_words

def get_Tensor(sentence, output_lang):
    output_tensor = tensorFromSentence(output_lang, sentence)
    output_tensor = pad_sequences(output_tensor, config['MAX_LENGTH'])
    output_length = output_tensor.size()[0]
    output_tensor = output_tensor.unsqueeze(0)
    return output_tensor

def calculateLoss(decoder, sentence, output_lang, p):
    decoder.eval()
    with torch.no_grad():
        criterion = nn.NLLLoss(ignore_index=0)
        output_tensor = get_Tensor(sentence, output_lang)

        decoder_hidden = decoder.initHidden(1)
        #encoder_hidden = encoder.initHidden(1, True)
        #print(input_tensor)
        c = 0
        a = []
        for i in output_tensor[0]:
            if i.item() == 0:
                a.append(0)
            else:
                c += 1
                a.append(1)
        output_tensor_len = torch.tensor([c], device=device)
        output_tensor_mask = get_mask(output_tensor_len)
        batch_size = 1
        decoder_input = output_tensor.narrow(1, 0, 1)
        #decoder_input = torch.full((batch_size, 1), SOS_token, device=device, dtype=torch.int64)
        decoded_words = []
        loss = 0
        #print(output_tensor)
        for di in range(output_tensor_len.item()-1):
            # -2 because of removing the SOS and EOS tokens
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, False)
            indices = torch.tensor([di+1], device=device)
            #print(output_tensor)
            target_t = torch.index_select(output_tensor, 1, indices)
            #print(target_t)
            #print(target_t.view(-1))
            loss += criterion(decoder_output[0], target_t.view(-1))
            '''topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])'''
            decoder_input = target_t
            #decoder_input = topi.squeeze(0).detach()
        #print(loss)
        #print(math.exp(loss))
        return loss.item()/(output_tensor_len.item()-1)
        # -2 for SOS and EOS token
        #return decoded_words, math.exp(loss)/max_length


def evaluatePerplexity(decoder, pairs, output_lang, p):
    candidate = []
    reference = []
    total_loss = 0
    n = min(1300, len(pairs))
    for i in range(n):
        pair = pairs[i]
        if p:
            print('>', pair)
        loss= calculateLoss(decoder, pair, output_lang, p)
        total_loss += loss
    print("Perplexity")
    print('%.4f ' % (math.exp(total_loss/n)))
    return math.exp(total_loss/n)

def evaluateSARI(complex_sentences, simple_sentences, generated_simple_sentences, p):
    total_score = 0
    n = len(complex_sentences)
    for i in range(n):
        score = calculate(complex_sentences[i], generated_simple_sentences[i], [simple_sentences[i]])
        total_score += score
    print(total_score/n)


def evaluateBLUE(decoder, pairs, output_lang, p):
    candidate = []
    reference = []
    n = min(1300, len(pairs))
    for i in range(n):
        pair = pairs[i]
        if p:
            print('>', pair[0])
            #print('=', pair[1])
        output_words = generate(decoder, pair[0], output_lang, p)
        #if p:
            #print([pair[1].split(' ')])
        #    print(output_words[:-1])
        candidate.append(output_words[:-1])
        reference.append([pair[0].split(' ')])
        output_sentence = ' '.join(output_words)
        if p:
            print('<', output_sentence)
            print('')
    '''if p:
        print(reference)
        print(candidate)'''
    print('Cumulative 4-gram: %f' % corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))
