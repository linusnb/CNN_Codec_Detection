	E??S??@E??S??@!E??S??@	:??kW??:??kW??!:??kW??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6E??S??@??>$@e@1??P1?8w@A?4?Ry;??I?8~?42@Y]m???{??*	L7?A???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2S@?? ?"@!*-#?X@)??B?"@1]????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@????}r??!?f???)????}r??1?f???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchw,?IEc??!??d????)w,?IEc??1??d????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn?r???!???~???)˄_??M??1-3?????:Preprocessing2F
Iterator::Model:̗`??!?j?v?w??)?LM?7?q?1~U???&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9:??kW??I?.)@@Q??ܰP?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??>$@e@??>$@e@!??>$@e@      ??!       "	??P1?8w@??P1?8w@!??P1?8w@*      ??!       2	?4?Ry;???4?Ry;??!?4?Ry;??:	?8~?42@?8~?42@!?8~?42@B      ??!       J	]m???{??]m???{??!]m???{??R      ??!       Z	]m???{??]m???{??!]m???{??b      ??!       JGPUY:??kW??b q?.)@@y??ܰP?P@