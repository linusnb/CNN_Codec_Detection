?	?ڥ箔@?ڥ箔@!?ڥ箔@      ??!       "n
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
	?K?F?c@?K?F?c@!?K?F?c@      ??!       "	??`??@??`??@!??`??@*      ??!       2	????I??????I??!????I??:	??%@??%@!??%@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??~??(@y?#?<??U@?"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?}?o@s??!?}?o@s??0"l
@gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?B??s>??!T?b)????0"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?^g???!c;]Q???0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??8"??!0?HP??0"j
?gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropInputConv2DBackpropInput?,W>?ު?!???6???0"=
sequential/conv_layer4/Relu_FusedConv2D????eb??!̝Wr\???"=
sequential/conv_layer3/Relu_FusedConv2D$?.???!??8?7F??"=
sequential/conv_layer2/Relu_FusedConv2Dc~5?nڧ?!??+?????"j
?gradient_tape/sequential/conv_layer4/Conv2D/Conv2DBackpropInputConv2DBackpropInput]??'æ??!j4??J>??0"\
;gradient_tape/sequential/maxpool_layer3/MaxPool/MaxPoolGradMaxPoolGrad???pK&??!Rr,W}???Q      Y@Yf??:D?2@a??L?nPT@q? ,
}8@y???+?C?"?

both?Your program is POTENTIALLY input-bound because 12.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?24.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 