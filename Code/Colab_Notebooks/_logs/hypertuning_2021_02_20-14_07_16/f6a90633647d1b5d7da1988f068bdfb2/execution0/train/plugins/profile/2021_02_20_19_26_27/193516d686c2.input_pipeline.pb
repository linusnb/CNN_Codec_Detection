	W??Y??@W??Y??@!W??Y??@	Z??x4??Z??x4??!Z??x4??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6W??Y??@S?'?ݳd@1??9\H?@A?˛õ??I~8H???@Yd> Й4??*	9??v?&?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?]??q"@!?/?0??X@)?P29?K"@1 !????X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?7k???!*ABy??)?7k???1*ABy??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch|??&??!@?1?????)|??&??1@?1?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism˻????!1??????){?ю~??1Cf?d$???:Preprocessing2F
Iterator::Modele???V??!?.??????)?D?e??v?10?f????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 10.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Z??x4??I??ߵ||$@Q??&?)mV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S?'?ݳd@S?'?ݳd@!S?'?ݳd@      ??!       "	??9\H?@??9\H?@!??9\H?@*      ??!       2	?˛õ???˛õ??!?˛õ??:	~8H???@~8H???@!~8H???@B      ??!       J	d> Й4??d> Й4??!d> Й4??R      ??!       Z	d> Й4??d> Й4??!d> Й4??b      ??!       JGPUYZ??x4??b q??ߵ||$@y??&?)mV@