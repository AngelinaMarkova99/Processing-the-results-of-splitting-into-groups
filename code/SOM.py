import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

warnings.filterwarnings('ignore')
tf.disable_v2_behavior()

class SOMNetwork():
	def __init__(self, input_dim, dim=10, sigma=None, learning_rate=0.1, tay2=1000, dtype=tf.float32):
		# В случае если сигма не определена
		if not sigma:
			sigma = dim / 2
		self.dtype = dtype
		# Определение констант для обучения
		self.dim = tf.constant(dim, dtype=tf.int64)
		self.learning_rate = tf.constant(learning_rate, dtype=dtype, name='learning_rate')
		self.sigma = tf.constant(sigma, dtype=dtype, name='sigma')
		self.tay1 = tf.constant(1000/np.log(sigma), dtype=dtype, name='tay1')
		self.minsigma = tf.constant(sigma * np.exp(-1000/(1000/np.log(sigma))), dtype=dtype, name='min_sigma')
		self.tay2 = tf.constant(tay2, dtype=dtype, name='tay2')
		# Входной вектор
		self.x = tf.placeholder(shape=[input_dim], dtype=dtype, name='input')
		# Номер итерации
		self.n = tf.placeholder(dtype=dtype, name='iteration')
		# Матрица весов
		self.w = tf.Variable(tf.random_uniform([dim*dim, input_dim], minval=-1, maxval=1, dtype=dtype),
			dtype=dtype, name='weights')
		# Матрица позиций всех нейронов, для определения расстояния между победившим и соседним нейронами
		self.positions = tf.where(tf.fill([dim, dim], True))

	# def feed(self, input):
	# 	init = tf.global_variables_initializer()
	# 	with tf.Session() as sess:
	# 		init.run()
	# 		win_index = sess.run(self.__competition(), feed_dict={self.x: input})
	# 		win_index_2d = np.array([win_index//self.dim.eval(), win_index-win_index//self.dim.eval()*self.dim.eval()])
	# 	return win_index_2d

	# Функция для создания операции процесса конкуренции
	def __competition(self, info=''):
		with tf.name_scope(info+'competition') as scope:
			# Вычисление евклидова расстояния
			distance = tf.sqrt(tf.reduce_sum(tf.square(self.x - self.w), axis=1))
		return tf.argmin(distance, axis=0)

	def training_op(self):
		# Определение индекса победившего нейрона
		win_index = self.__competition('train_')
		with tf.name_scope('cooperation') as scope:
			coop_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(self.positions -
				[win_index//self.dim, win_index-win_index//self.dim*self.dim],
				dtype=self.dtype)), axis=1))
			sigma = tf.cond(self.n > 1000, lambda: self.minsigma, lambda: self.sigma * tf.exp(-self.n/self.tay1))
			sigma_summary = tf.summary.scalar('Sigma', sigma)
			tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(sigma))) # topological neighbourhood
		with tf.name_scope('adaptation') as scope:
			lr = self.learning_rate * tf.exp(-self.n/self.tay2)
			minlr = tf.constant(0.01, dtype=self.dtype, name='min_learning_rate')
			lr = tf.cond(lr <= minlr, lambda: minlr, lambda: lr)
			lr_summary = tf.summary.scalar('Learning rate', lr)
			delta = tf.transpose(lr * tnh * tf.transpose(self.x - self.w))
			training_op = tf.assign(self.w, self.w + delta)
		return training_op, lr_summary, sigma_summary


#== Test SOM Network ==

def test_som_with_color_data():
	# Размер сети som_dim x som_dim
	som_dim = 20
	som = SOMNetwork(input_dim=3, dim=som_dim, dtype=tf.float64, sigma=3)
	test_data = np.random.uniform(0, 1, (250000, 3))
	print(test_data)
	# Основной цикл обучения и вывод результатов
	training_op, lr_summary, sigma_summary = som.training_op()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		init.run()
		img1 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
		print(img1)
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(img1)
		start = time.time()
		for i, color_data in enumerate(test_data):
			if i % 1000 == 0:
				print('Номер итерации:', i)
			sess.run(training_op, feed_dict={som.x: color_data, som.n: i})
		end = time.time()
		print('Затраченное время:', end - start)
		img2 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
		plt.subplot(122)
		plt.imshow(img2)
	plt.show()

# test_som_with_color_data()

data = pd.read_csv('./supermarket_sales - Sheet1.csv')
# Вывод базы данных
# print(data)
df = data.copy()

# Какую категорию продуктов чаще всего приобретают
# plt.figure(figsize=(10, 5))
# # x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # plt.grid()
# # sns.barplot(y=df['Product line'], x=df.Rating, palette='magma',  order=["Food and beverages", "Fashion accessories", "Health and beauty", "Electronic accessories", "Sports and travel", "Home and lifestyle"])
# # plt.gca().set(xlabel='Рейтинг', ylabel='Категория продуктов', title='Рейтинг продуктов', facecolor='grey')
# # plt.xticks(x_ticks)
# # plt.show()
# #
# # plt.figure(figsize=(10, 5))
# # sns.countplot(y='Product line', hue="City", data=df, palette='magma')
# # plt.gca().set(xlabel='Количество', ylabel='Категория продуктов', facecolor='grey')
# # plt.show()

var_city = df['City'].copy()
var_product = df['Product line'].copy()

for i in range(len(var_city)):
	if var_city[i] == 'Mandalay':
		var_city[i] = 0
	if var_city[i] == 'Naypyitaw':
		var_city[i] = 0.5
	if var_city[i] == 'Yangon':
		var_city[i] = 1

for i in range(len(var_product)):
	if var_product[i] == 'Food and beverages':
		var_product[i] = 0
	if var_product[i] == 'Fashion accessories':
		var_product[i] = 0.2
	if var_product[i] == 'Health and beauty':
		var_product[i] = 0.4
	if var_product[i] == 'Electronic accessories':
		var_product[i] = 0.6
	if var_product[i] == 'Sports and travel':
		var_product[i] = 0.8
	if var_product[i] == 'Home and lifestyle':
		var_product[i] = 1

test = pd.DataFrame({'City': var_city, 'Product line': var_product, 'Rating': df['Rating']/10})

def test_som_with_supermarket_data():
	som_dim = 20
	som = SOMNetwork(input_dim=3, dim=som_dim, dtype=tf.float64, sigma=3)
	test_data = np.array(test)
	print(test_data)
	training_op, lr_summary, sigma_summary = som.training_op()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		init.run()
		img1 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
		print(img1)
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(img1)
		start = time.time()
		for i, color_data in enumerate(test_data):
			if i % 100 == 0:
				print('Номер итерации:', i)
			sess.run(training_op, feed_dict={som.x: color_data, som.n: i})
		end = time.time()
		print('Затраченное время:', end - start)
		img2 = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
		plt.subplot(122)
		plt.imshow(img2)
	plt.show()

test_som_with_supermarket_data()
