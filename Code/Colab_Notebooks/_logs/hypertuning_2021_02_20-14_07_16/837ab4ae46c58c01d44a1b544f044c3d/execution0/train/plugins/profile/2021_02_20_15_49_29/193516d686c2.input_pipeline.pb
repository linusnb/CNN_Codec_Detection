	9?Cm?r@9?Cm?r@!9?Cm?r@	??<?????<???!??<???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69?Cm?r@?L?*e@1?\pN^@AX?B?_˫?I?[??b?@Y=~oӟ??*	~?5^j??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?聏?
$@!e??B??X@)S?k%t?#@1??.??X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?'????!??U???)?'????1??U???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?N?Z?7??!~???9??)?N?Z?7??1~???9??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!}@??????)?9??q???1??Ġ??:Preprocessing2F
Iterator::Modelbi?G5???!yM??ި??)y:W??u?1?ϐ-<???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??<???I???r}|M@Q??*Q?kD@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?L?*e@?L?*e@!?L?*e@      ??!       "	?\pN^@?\pN^@!?\pN^@*      ??!       2	X?B?_˫?X?B?_˫?!X?B?_˫?:	?[??b?@?[??b?@!?[??b?@B      ??!       J	=~oӟ??=~oӟ??!=~oӟ??R      ??!       Z	=~oӟ??=~oӟ??!=~oӟ??b      ??!       JGPUY??<???b q???r}|M@y??*Q?kD@