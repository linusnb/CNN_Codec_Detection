	G?I?ȓ@G?I?ȓ@!G?I?ȓ@	F7?b1???F7?b1???!F7?b1???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6G?I?ȓ@??'??Re@1?8???@A-]?6?ɺ?IPoFͷ@Yn?????*	???#??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2W'g(??"@!0
	?D?X@)M-[???"@1UW4<?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?[?O???!??=g? ??)</O犲?1j???1??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@F	?=b??!?mYj\???)F	?=b??1?mYj\???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch}??A?<??!(??џ??)}??A?<??1(??џ??:Preprocessing2F
Iterator::Modelu?????! t??????)?Nw?x?v?1! ??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9F7?b1???I@Pr??+@Q?e??>?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??'??Re@??'??Re@!??'??Re@      ??!       "	?8???@?8???@!?8???@*      ??!       2	-]?6?ɺ?-]?6?ɺ?!-]?6?ɺ?:	PoFͷ@PoFͷ@!PoFͷ@B      ??!       J	n?????n?????!n?????R      ??!       Z	n?????n?????!n?????b      ??!       JGPUYF7?b1???b q@Pr??+@y?e??>?U@