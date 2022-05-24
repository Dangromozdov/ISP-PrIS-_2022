import telebot
import onnxruntime
import sentencepiece as spm
import numpy as np
import pickle
import re
import random
from telebot import types

flag = 0


file_w = open('wisdom.txt', 'r', encoding='UTF-8')
wisdom = file_w.read().split('\n')
file_w.close()

stic = open('Stickers/marx.webp', 'rb')
sti = stic.read()
stic.close()

sess_enc = onnxruntime.InferenceSession('seq2seq_enc.onnx') 
sess_dec = onnxruntime.InferenceSession('seq2seq_dec.onnx') 
with open('vocabs.pickle', 'rb') as f: 
	SRC_SOS, SRC_EOS, SRC_STOI, TRG_SOS, TRG_EOS, TRG_STOI, TRG_ITOS = pickle.load(f)

bot = telebot.TeleBot('5351158031:AAE4cOdUdpbgPBHiPOBS2ErLDViwvQzL1D8')

def tokenize(text):
	return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",text)


def preprocess(text):
	tokens = [t.lower() for t in tokenize(text)]
	tokens = [SRC_SOS] + tokens + [SRC_EOS]
	src_indexes = [SRC_STOI.get(token, 0) for token in tokens]
	src_tensor = np.int64(src_indexes).reshape(1, -1)
	src_mask = (np.int64(src_indexes) != 1).reshape(1, 1, 1, -1)
	return src_tensor, src_mask


def get_trg_mask(trg_tensor):
	trg_pad_mask = (trg_tensor != 1).reshape(1, 1, 1, -1)
	trg_len = trg_tensor.shape[1]
	trg_sub_mask = np.tril(np.ones((trg_len, trg_len), dtype=np.bool))
	return trg_pad_mask & trg_sub_mask
    
def Translate(message):

	test_text = message.text
	print(test_text)
	src_tensor, src_mask = preprocess(test_text)
	enc_src = sess_enc.run(None, {'src_tensor': src_tensor,
                              'src_mask': src_mask})[0]
	trg_indexes = [TRG_STOI[TRG_SOS]]
	for i in range(128):
		trg_tensor = np.int64(trg_indexes).reshape(1, -1)
		trg_mask = get_trg_mask(trg_tensor)
		output, attention = sess_dec.run(None, {'trg_tensor': trg_tensor, 
                                            'enc_src': enc_src,
                                            'trg_mask': trg_mask,
                                            'src_mask': src_mask})
		pred_token = output.argmax(axis=2)[:,-1].item()
		print(pred_token)
		trg_indexes.append(pred_token)
		if pred_token == TRG_STOI[TRG_EOS]:
			break

	trg_tokens = [TRG_ITOS[i] for i in trg_indexes]
	' '.join(trg_tokens[1:-1])
	result = ' '
	for i in range(len(trg_tokens[1:-1])):
		result = result + ' '+ trg_tokens[i+1]
	bot.send_message(message.from_user.id, result)

@bot.message_handler(commands=["start"])
def start(m, res=False):
	markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
	item1=types.KeyboardButton("Цитата")
	markup.add(item1)
	bot.send_message(m.chat.id, "Hallo! Этот бот предназначен для машинного перевода с немецкого на итальянский. Больше информации: /help \nДля получения случайной цитаты на немецком нажмите кнопку 'Цитата'", reply_markup=markup)
    
@bot.message_handler(content_types=['text'])

def get_text_messages(message):
	global flag
	if flag == 0:
		#if message.text == "/start":
			#bot.send_message(message.from_user.id, "Hallo! Этот бот предназначен для машинного перевода с немецкого на итальянский. Больше информации: /help")
		if message.text == "/help":
			bot.send_message(message.from_user.id, "Чтобы перевести текст с немецкого на итальянский отправьте команду: /translator")
		elif message.text == "/translator":
			bot.send_message(message.from_user.id, "Напишите текст на немецком, и я его переведу: ")
			flag = 2
		elif message.text.strip()=="Цитата":
			answer = random.choice(wisdom)
			bot.send_message(message.chat.id, answer) 
			bot.send_sticker(message.chat.id, sti)     
		else:
			bot.send_message(message.from_user.id, "Не понимаю, что Вам нужно. Воспользуйтесь командой помощи: /help")
	elif flag==2:
		Translate(message)
		flag=0


bot.polling(none_stop=True, interval=0)