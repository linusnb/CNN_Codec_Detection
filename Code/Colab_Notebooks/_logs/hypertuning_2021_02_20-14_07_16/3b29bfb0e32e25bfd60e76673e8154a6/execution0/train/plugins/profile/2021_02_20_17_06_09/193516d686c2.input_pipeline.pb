	k?K??@k?K??@!k?K??@	"????s??"????s??!"????s??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6k?K??@T???~?e@1???U?!}@A?6??nf??I??	j8
@Yr??9???*	??ʡ???@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2E,b?	$@! ??p#?X@)u?V?#@1? Z?V?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???	/???!?>?Cc???)???	/???1?>?Cc???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?T?=ϟ??!nH?????)?T?=ϟ??1nH?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`YiR
???!?Z?????)?]gEԤ?1m??p???:Preprocessing2F
Iterator::Model???????!?1?Gn??)?uT5At?1?M????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9!????s??IHwO?	Y;@Qk?'??R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T???~?e@T???~?e@!T???~?e@      ??!       "	???U?!}@???U?!}@!???U?!}@*      ??!       2	?6??nf???6??nf??!?6??nf??:	??	j8
@??	j8
@!??	j8
@B      ??!       J	r??9???r??9???!r??9???R      ??!       Z	r??9???r??9???!r??9???b      ??!       JGPUY!????s??b qHwO?	Y;@yk?'??R@