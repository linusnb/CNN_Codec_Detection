?	?˚X0??@?˚X0??@!?˚X0??@	ԅi??v??ԅi??v??!ԅi??v??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?˚X0??@m9??*?e@1y7ې@A4iSu?l??I?ڧ?1? @Y?R	O????*	$??~j??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?8?	??!@! Gp???X@)?)??a!@1ao9{?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???C???!j?k7???)???C???1j?k7???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch^M??????!???$????)^M??????1???$????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism@1?d????!???+U??)E*?-9??1+t?$??:Preprocessing2F
Iterator::ModelV???5??!???>1??)c???Ju?1?1?^1???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ԅi??v??I? 链?+@Q,+)?,~U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m9??*?e@m9??*?e@!m9??*?e@      ??!       "	y7ې@y7ې@!y7ې@*      ??!       2	4iSu?l??4iSu?l??!4iSu?l??:	?ڧ?1? @?ڧ?1? @!?ڧ?1? @B      ??!       J	?R	O?????R	O????!?R	O????R      ??!       Z	?R	O?????R	O????!?R	O????b      ??!       JGPUYԅi??v??b q? 链?+@y,+)?,~U@?"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterg?xT????!g?xT????0"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?2iw??!???^R???0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?1???@??!??!(Q&??0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput????????!?_;??!??0"=
sequential/conv_layer3/Relu_FusedConv2D)f??????!fc8???"=
sequential/conv_layer2/Relu_FusedConv2D??@?Y??!-?d\????"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInputb{??????!?(??d???0"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInput$?c???!$˾.@???0"=
sequential/conv_layer4/Relu_FusedConv2D?x?ʌ???!?v?|o??"\
;gradient_tape/sequential/maxpool_layer2/MaxPool/MaxPoolGradMaxPoolGradd??????!-%\??Q      Y@YO+????2@a-5???^T@q??:B&N)@y8???&^B?"?

both?Your program is POTENTIALLY input-bound because 13.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?12.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 