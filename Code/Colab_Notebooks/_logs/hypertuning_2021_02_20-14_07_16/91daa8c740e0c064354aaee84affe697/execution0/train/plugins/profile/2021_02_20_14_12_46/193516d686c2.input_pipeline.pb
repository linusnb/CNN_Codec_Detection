	??.Q??z@??.Q??z@!??.Q??z@	?j:??t???j:??t??!?j:??t??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??.Q??z@???b@12>?^?wq@A?SH?9??IG???8@Yw/??Q???*	֣p=???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2\??AAA#@!?????X@)it?3#@1l?i-??X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?yq???!?	?FId??)?yq???1?	?FId??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchH?ξ? ??!????<??)H?ξ? ??1????<??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!R̰2?D??)???:U???1?͉?ۚ??:Preprocessing2F
Iterator::Model?M?a????!??? $??)?z6?>w?1?H?M???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?j:??t??Ip???YA@Q???|LP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???b@???b@!???b@      ??!       "	2>?^?wq@2>?^?wq@!2>?^?wq@*      ??!       2	?SH?9???SH?9??!?SH?9??:	G???8@G???8@!G???8@B      ??!       J	w/??Q???w/??Q???!w/??Q???R      ??!       Z	w/??Q???w/??Q???!w/??Q???b      ??!       JGPUY?j:??t??b qp???YA@y???|LP@