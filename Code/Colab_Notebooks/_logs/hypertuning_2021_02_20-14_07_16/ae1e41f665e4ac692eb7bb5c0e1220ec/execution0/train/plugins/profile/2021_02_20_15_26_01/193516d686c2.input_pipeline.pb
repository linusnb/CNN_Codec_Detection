	?????/s@?????/s@!?????/s@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?????/s@W%?}/e@1??kт?`@A?W?\T??I???v?3@*	w??/%?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??5!?q"@!D{?6??X@)ß??L"@1??h?]?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@\u?)ɲ?!?CK?F??)\u?)ɲ?1?CK?F??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch:??H???!???d?B??):??H???1???d?B??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?nK??3??!I%q=d???)ٕ??zO??1K?,<???:Preprocessing2F
Iterator::Model???~??!?]B?d???)??T???t?1??-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???Y}fL@Q^H???E@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W%?}/e@W%?}/e@!W%?}/e@      ??!       "	??kт?`@??kт?`@!??kт?`@*      ??!       2	?W?\T???W?\T??!?W?\T??:	???v?3@???v?3@!???v?3@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???Y}fL@y^H???E@