?	??OFVz@??OFVz@!??OFVz@	?@??n]???@??n]??!?@??n]??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??OFVz@?X"??_@1?Z??r@A%?S;?Զ?IO#-???@YU/??d???*	?x?&?1?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??E;L#@!???PT@)?bc^G$#@1??Q??%T@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismȗP???@!???$?2@)ٙB?5V@1?t????2@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@3??????!y#???)3??????1y#???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?{?????!D?U??b??)?{?????1D?U??b??:Preprocessing2F
Iterator::Modelͱ???@!?a????2@)	4??yt?1Qc+{???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 30.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?@??n]??Ih?>???>@Q%?o?b3Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?X"??_@?X"??_@!?X"??_@      ??!       "	?Z??r@?Z??r@!?Z??r@*      ??!       2	%?S;?Զ?%?S;?Զ?!%?S;?Զ?:	O#-???@O#-???@!O#-???@B      ??!       J	U/??d???U/??d???!U/??d???R      ??!       Z	U/??d???U/??d???!U/??d???b      ??!       JGPUY?@??n]??b qh?>???>@y%?o?b3Q@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterBйCD`??!BйCD`??0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?6?L|??!???G5???0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput좧5????!^sT?;??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterY(!o?|??!t??S???0"=
sequential/conv_layer2/Relu_FusedConv2D????????!3|?q??"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInputp???þ??!??1B???0"=
sequential/conv_layer3/Relu_FusedConv2Da???ᴭ?!FA?a????"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInputج?q??!<?{?a??0"=
sequential/conv_layer4/Relu_FusedConv2D?,??;פ?!??4(???"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??c????!??{????0Q      Y@YZT??5?<@a?????Q@qH?"??(@y磺 ?[a?"?

both?Your program is POTENTIALLY input-bound because 30.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?12.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 