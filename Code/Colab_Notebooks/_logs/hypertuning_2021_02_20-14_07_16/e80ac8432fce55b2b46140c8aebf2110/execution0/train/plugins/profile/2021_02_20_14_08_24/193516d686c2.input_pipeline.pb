	??OFVz@??OFVz@!??OFVz@	?@??n]???@??n]??!?@??n]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??OFVz@?X"??_@1?Z??r@A%?S;?Զ?IO#-???@YU/??d???*	?x?&?1?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??E;L#@!???PT@)?bc^G$#@1??Q??%T@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismȗP???@!???$?2@)ٙB?5V@1?t????2@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@3??????!y#???)3??????1y#???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?{?????!D?U??b??)?{?????1D?U??b??:Preprocessing2F
Iterator::Modelͱ???@!?a????2@)	4??yt?1Qc+{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?@??n]??Ih?>???>@Q%?o?b3Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?X"??_@?X"??_@!?X"??_@      ??!       "	?Z??r@?Z??r@!?Z??r@*      ??!       2	%?S;?Զ?%?S;?Զ?!%?S;?Զ?:	O#-???@O#-???@!O#-???@B      ??!       J	U/??d???U/??d???!U/??d???R      ??!       Z	U/??d???U/??d???!U/??d???b      ??!       JGPUY?@??n]??b qh?>???>@y%?o?b3Q@