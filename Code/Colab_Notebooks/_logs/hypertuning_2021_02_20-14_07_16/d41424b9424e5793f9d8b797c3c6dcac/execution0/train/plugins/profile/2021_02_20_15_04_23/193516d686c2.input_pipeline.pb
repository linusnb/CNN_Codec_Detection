	?#0`c?@?#0`c?@!?#0`c?@	'?ǯ?k?'?ǯ?k?!'?ǯ?k?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?#0`c?@??????c@1&VF#?Е@A???t ???IP?R)?@Y9?⪲???*	?V??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????#@!}/{??X@)?????"@1rq???X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@{???`Ĳ?!>?Ȍ???){???`Ĳ?1>?Ȍ???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch%?}?e???!??????)%?}?e???1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????%??!?zc?+???)i???>Ȓ?1~?m?????:Preprocessing2F
Iterator::Model!??i??!AhB?q??)J??{dsu?1tL???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'?ǯ?k?I??x}?%@Q?]?V?\V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????c@??????c@!??????c@      ??!       "	&VF#?Е@&VF#?Е@!&VF#?Е@*      ??!       2	???t ??????t ???!???t ???:	P?R)?@P?R)?@!P?R)?@B      ??!       J	9?⪲???9?⪲???!9?⪲???R      ??!       Z	9?⪲???9?⪲???!9?⪲???b      ??!       JGPUY'?ǯ?k?b q??x}?%@y?]?V?\V@