	@?:s?*?@@?:s?*?@!@?:s?*?@	?a??????a?????!?a?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6@?:s?*?@?.??e@1???@?:w@AOu??p??I9{???@Y?MI??*	?O???R?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?hq?0?"@!?a???X@)v?ݑ?z"@1?A?e?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???I????!p?>?P??)???I????1p?>?P??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??u?+.??!oJ??ۍ??)??u?+.??1oJ??ۍ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism%?}?e???!?T/`dS??)E?
)????1??j??1??:Preprocessing2F
Iterator::ModelT5A?} ??!?O?????)?B:<??s?1???Ԓ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 31.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?a?????I?????@@Q??s??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.??e@?.??e@!?.??e@      ??!       "	???@?:w@???@?:w@!???@?:w@*      ??!       2	Ou??p??Ou??p??!Ou??p??:	9{???@9{???@!9{???@B      ??!       J	?MI???MI??!?MI??R      ??!       Z	?MI???MI??!?MI??b      ??!       JGPUY?a?????b q?????@@y??s??P@