import tensorflow as tf


a = [b'26:21.0,4.5,2009-06-15 17:26:21 UTC,-73.844311,40.721319,-73.84161,40.712278,1']
b = tf.strings.split(a,",")

print(b.shape)
print(type(b))
print(type(b[0,1]))
print( str(b[0,1].numpy(), "utf-8" ))
print(type(a))
a_ts = tf.convert_to_tensor(a)
print(type(a_ts))
print(a_ts.numpy())
print(b.values)


cd = tf.constant("ley")
print(cd)
print(type(cd))

#bt = b.numpy()
# b_list = b.to_list()
# for x in b_list:
#     for y in x:
#         print(type(y))
#         print(str(y, "utf-8"))
#         print(type(y))
#
# print(b_list[0:1, ])
