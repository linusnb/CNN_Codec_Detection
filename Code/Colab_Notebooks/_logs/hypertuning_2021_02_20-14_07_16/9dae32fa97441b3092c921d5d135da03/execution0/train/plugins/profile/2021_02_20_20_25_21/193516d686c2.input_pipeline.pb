	a?9'x@a?9'x@!a?9'x@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-a?9'x@R??c@1?????k@A5{????I?qS-@*	?Zdt?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?`8?0?"@!???J?X@)ެ????"@1Z38??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismu???????!;???3???)F?T?=ϯ?1?v?
??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?o?4(???!h??????)?o?4(???1h??????:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???vۅ??!\]?5?0??)???vۅ??1\]?5?0??:Preprocessing2F
Iterator::Model?ص?ݒ??!?zWQ???)C??w?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 40.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?	p?]E@Q???h?L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	R??c@R??c@!R??c@      ??!       "	?????k@?????k@!?????k@*      ??!       2	5{????5{????!5{????:	?qS-@?qS-@!?qS-@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?	p?]E@y???h?L@