	??s@??s@!??s@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??s@?i?WV?c@1	Q????a@A{?Fw;??I??qn.@*	/?$f??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?ݑ??? @!????`?X@)m;m?? @1n????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?v?$$Ү?!?@????)?v?$$Ү?1?@????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?wG?j???!?????)?wG?j???1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismn?2d???!??ĺ??)?>;??b??1??Ǜc??:Preprocessing2F
Iterator::Model???5????!7=?????)??֦??v?1?o4?4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??
U1?J@Q

???G@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?i?WV?c@?i?WV?c@!?i?WV?c@      ??!       "		Q????a@	Q????a@!	Q????a@*      ??!       2	{?Fw;??{?Fw;??!{?Fw;??:	??qn.@??qn.@!??qn.@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??
U1?J@y

???G@