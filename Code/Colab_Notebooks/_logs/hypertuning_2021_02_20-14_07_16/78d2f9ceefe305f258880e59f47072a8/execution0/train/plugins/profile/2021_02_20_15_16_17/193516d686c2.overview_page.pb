?	G?I?ȓ@G?I?ȓ@!G?I?ȓ@	F7?b1???F7?b1???!F7?b1???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6G?I?ȓ@??'??Re@1?8???@A-]?6?ɺ?IPoFͷ@Yn?????*	???#??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2W'g(??"@!0
	?D?X@)M-[???"@1UW4<?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?[?O???!??=g? ??)</O犲?1j???1??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@F	?=b??!?mYj\???)F	?=b??1?mYj\???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch}??A?<??!(??џ??)}??A?<??1(??џ??:Preprocessing2F
Iterator::Modelu?????! t??????)?Nw?x?v?1! ??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9F7?b1???I@Pr??+@Q?e??>?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??'??Re@??'??Re@!??'??Re@      ??!       "	?8???@?8???@!?8???@*      ??!       2	-]?6?ɺ?-]?6?ɺ?!-]?6?ɺ?:	PoFͷ@PoFͷ@!PoFͷ@B      ??!       J	n?????n?????!n?????R      ??!       Z	n?????n?????!n?????b      ??!       JGPUYF7?b1???b q@Pr??+@y?e??>?U@?"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]??n-???!]??n-???0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!͛?JO??0"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterՃW7OV??!a>1?Y???0"=
sequential/conv_layer4/Relu_FusedConv2D!J??7??!????L???"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInput???_~{??!o??????0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput#U?????!a=MP???0"=
sequential/conv_layer3/Relu_FusedConv2D.??#?J??!?H????"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputq?? ğ?!?M???0"=
sequential/conv_layer2/Relu_FusedConv2D??V2??!???????"\
;gradient_tape/sequential/maxpool_layer3/MaxPool/MaxPoolGradMaxPoolGrad	????t??!????Lg??Q      Y@Y????1@aC
=x?T@qK?v??1@y?[????A?"?

both?Your program is POTENTIALLY input-bound because 13.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?17.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 