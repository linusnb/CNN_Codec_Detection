	(
􉼦p@(
􉼦p@!(
􉼦p@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-(
􉼦p@??y??a@1??]??(^@A??ɍ"k??I?ic@*	????C?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??9}?$@!?8???X@)??V`Ȃ$@1?[?ǵX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?@?ش??!Iy?SG??)?@?ش??1Iy?SG??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchp"???ӧ?!??^?δ??)p"???ӧ?1??^?δ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism? ????!4y??M??)?=$|?o??1?]T?m???:Preprocessing2F
Iterator::Model?ܙ	?s??!ͩ?c)??)??;???v?1?	#?j???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???H\K@Q	S?i??F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??y??a@??y??a@!??y??a@      ??!       "	??]??(^@??]??(^@!??]??(^@*      ??!       2	??ɍ"k????ɍ"k??!??ɍ"k??:	?ic@?ic@!?ic@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???H\K@y	S?i??F@