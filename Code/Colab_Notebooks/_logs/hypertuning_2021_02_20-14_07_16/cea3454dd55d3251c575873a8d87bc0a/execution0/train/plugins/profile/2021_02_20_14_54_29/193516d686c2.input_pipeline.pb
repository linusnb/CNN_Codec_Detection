	???B??~@???B??~@!???B??~@	|*<v???|*<v???!|*<v???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???B??~@??"j??d@1c?T4?4t@A???~????I?_????@Y??~????*	??n?q?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2j/?혺"@!??6?=?X@)M???"@18?5?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?\?????!?V/C???)?\?????1?V/C???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchEկt><??!?L?P??)Eկt><??1?L?P??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismX??C???!J??u=??)?rg&Ε?1Q]-????:Preprocessing2F
Iterator::Model??o?N??!?$??!???)v?r??s?1f?i?:??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|*<v???I??????A@Q	?CȕMP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??"j??d@??"j??d@!??"j??d@      ??!       "	c?T4?4t@c?T4?4t@!c?T4?4t@*      ??!       2	???~???????~????!???~????:	?_????@?_????@!?_????@B      ??!       J	??~??????~????!??~????R      ??!       Z	??~??????~????!??~????b      ??!       JGPUY|*<v???b q??????A@y	?CȕMP@