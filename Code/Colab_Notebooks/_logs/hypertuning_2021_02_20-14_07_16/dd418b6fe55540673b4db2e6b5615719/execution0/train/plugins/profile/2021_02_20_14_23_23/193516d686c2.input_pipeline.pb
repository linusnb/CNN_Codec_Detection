	??H?y@??H?y@!??H?y@	?Nn?6????Nn?6???!?Nn?6???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??H?y@???_ac@1?뉮?Ln@A?|????I???a?
@Y\Va3???*	?&1???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?0??B6!@!=?yHL?X@)-&6?!@1????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?
b?k??!?M?;%???)?
b?k??1?M?;%???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ƃ-v???!?????>??)?ƃ-v???1?????>??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism1??c???!??.????)??^D?1??1S;????:Preprocessing2F
Iterator::Model???B????!ca)??Y??)????u?q?1?DHYu??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 38.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Nn?6???I`????C@QR?Iđ-N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???_ac@???_ac@!???_ac@      ??!       "	?뉮?Ln@?뉮?Ln@!?뉮?Ln@*      ??!       2	?|?????|????!?|????:	???a?
@???a?
@!???a?
@B      ??!       J	\Va3???\Va3???!\Va3???R      ??!       Z	\Va3???\Va3???!\Va3???b      ??!       JGPUY?Nn?6???b q`????C@yR?Iđ-N@