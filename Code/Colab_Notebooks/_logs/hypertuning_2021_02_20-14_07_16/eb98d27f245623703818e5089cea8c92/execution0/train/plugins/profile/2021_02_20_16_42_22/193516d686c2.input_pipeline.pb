	?ڥ箔@?ڥ箔@!?ڥ箔@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?ڥ箔@?K?F?c@1??`??@A????I??I??%@*	0?$?\?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?+?j?!@!cQ,??X@)?*??<j!@1?wI??|X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?#?&ݖ??!?????S??)?#?&ݖ??1?????S??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?鲘?|??!????l??)5bf??(??1	?\H???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetcheq???Щ?!?Aa?&??)eq???Щ?1?Aa?&??:Preprocessing2F
Iterator::Model?@?C???!. ?????)in??Kx?193ẁ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??~??(@Q?#?<??U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K?F?c@?K?F?c@!?K?F?c@      ??!       "	??`??@??`??@!??`??@*      ??!       2	????I??????I??!????I??:	??%@??%@!??%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??~??(@y?#?<??U@