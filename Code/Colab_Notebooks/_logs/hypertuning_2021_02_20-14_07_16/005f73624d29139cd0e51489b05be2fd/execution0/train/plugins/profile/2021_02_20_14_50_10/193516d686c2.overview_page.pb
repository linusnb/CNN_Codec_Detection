?	ސFN4{@ސFN4{@!ސFN4{@	h?g??c??h?g??c??!h?g??c??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ސFN4{@?M?=2?d@1??oD??p@AC7?????I??[?n+
@Y}?%?/??*	#??~J??@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??6?0!@!?훣d?X@)*t^c?!@1???sy?X@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@???i??!?1?ї???)???i??1?1?ї???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch_ѭ?????!???U????)_ѭ?????1???U????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??L?^???!`?[}????)Kw?ِ??1ψ?N?7??:Preprocessing2F
Iterator::ModelE?>?'I??!0	2????).?u?1 ??J???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9h?g??c??Iz???WC@Q?"&??N@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?M?=2?d@?M?=2?d@!?M?=2?d@      ??!       "	??oD??p@??oD??p@!??oD??p@*      ??!       2	C7?????C7?????!C7?????:	??[?n+
@??[?n+
@!??[?n+
@B      ??!       J	}?%?/??}?%?/??!}?%?/??R      ??!       Z	}?%?/??}?%?/??!}?%?/??b      ??!       JGPUYh?g??c??b qz???WC@y?"&??N@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??o????!??o????0"=
sequential/conv_layer2/Relu_FusedConv2D	?4?\E??!Ni-??L??"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterZ(??%??!?~B#????0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput????c??!%yF?????0"=
sequential/conv_layer3/Relu_FusedConv2D?)?Ά??!k?P\b???"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?f?$??!?Q?????0"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGradܣ??M#??!,R?L??"\
;gradient_tape/sequential/maxpool_layer2/MaxPool/MaxPoolGradMaxPoolGradB<i???!ۿ>}??"K
-gradient_tape/sequential/conv_layer1/ReluGradReluGrad???9??!a?ܐ??"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteroM:????!?G̨???0Q      Y@Y]蚼}?@@aҋ?!??P@q?p?p"Z+@y[?qxVd?"?

both?Your program is POTENTIALLY input-bound because 37.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 