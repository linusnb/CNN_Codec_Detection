	Wj1??@Wj1??@!Wj1??@	?!J?????!J????!?!J????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Wj1??@???w?uf@1^??#?Ut@A?ut\???I??	?yK@Y?7h?>???*	??x????@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?d8??8"@!?qC?X@);?5Y?"@1??WՖX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@H4?"??!???۶??)H4?"??1???۶??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchR||Bvަ?!1i??A$??)R||Bvަ?11i??A$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismR臭????!R??҉???)?Į?햔?1??1N?	??:Preprocessing2F
Iterator::Model????????!?pGx???)si??+In?1?<A瞤?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?!J????I?]?p?A@Q?^wC?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???w?uf@???w?uf@!???w?uf@      ??!       "	^??#?Ut@^??#?Ut@!^??#?Ut@*      ??!       2	?ut\????ut\???!?ut\???:	??	?yK@??	?yK@!??	?yK@B      ??!       J	?7h?>????7h?>???!?7h?>???R      ??!       Z	?7h?>????7h?>???!?7h?>???b      ??!       JGPUY?!J????b q?]?p?A@y?^wC?O@