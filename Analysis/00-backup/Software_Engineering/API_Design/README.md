# API设计 (API Design)

## 概述

API设计是构建应用程序接口的艺术和科学，涉及定义服务间通信的协议、数据格式和交互模式。
良好的API设计能够提高系统的可维护性、可扩展性和用户体验。

## API设计原则

### 1. RESTful API设计

```rust
// REST API资源定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub email: String,
    pub created_at: DateTime<Utc>,
}

// REST API路由设计
pub struct UserApi {
    user_service: UserService,
}

impl UserApi {
    // GET /users - 获取用户列表
    pub async fn get_users(&self, query: UserQuery) -> Result<Vec<User>, ApiError> {
        let users = self.user_service.get_users(&query).await?;
        Ok(users)
    }
    
    // GET /users/{id} - 获取单个用户
    pub async fn get_user(&self, id: &str) -> Result<User, ApiError> {
        let user = self.user_service.get_user(id).await?;
        Ok(user)
    }
    
    // POST /users - 创建用户
    pub async fn create_user(&self, user_data: CreateUserRequest) -> Result<User, ApiError> {
        let user = self.user_service.create_user(&user_data).await?;
        Ok(user)
    }
    
    // PUT /users/{id} - 更新用户
    pub async fn update_user(&self, id: &str, user_data: UpdateUserRequest) -> Result<User, ApiError> {
        let user = self.user_service.update_user(id, &user_data).await?;
        Ok(user)
    }
    
    // DELETE /users/{id} - 删除用户
    pub async fn delete_user(&self, id: &str) -> Result<(), ApiError> {
        self.user_service.delete_user(id).await?;
        Ok(())
    }
}
```

### 2. HTTP状态码使用

```rust
// API响应结构
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub message: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            message: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn error(error: ApiError) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            message: Some(error.message.clone()),
            timestamp: Utc::now(),
        }
    }
}
```

### 3. 错误处理

```rust
// API错误定义
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

impl ApiError {
    pub fn not_found(resource: &str) -> Self {
        Self {
            code: "NOT_FOUND".to_string(),
            message: format!("{} not found", resource),
            details: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn validation_error(field: &str, message: &str) -> Self {
        Self {
            code: "VALIDATION_ERROR".to_string(),
            message: format!("Validation error for field '{}': {}", field, message),
            details: None,
            timestamp: Utc::now(),
        }
    }
}
```

## GraphQL API设计

### 1. Schema定义

```rust
// GraphQL Schema
pub struct UserSchema;

#[Object]
impl UserSchema {
    // 查询用户
    async fn user(&self, ctx: &Context<'_>, id: String) -> Result<Option<User>, Error> {
        let user_service = ctx.data::<UserService>()?;
        let user = user_service.get_user(&id).await?;
        Ok(user)
    }
    
    // 查询用户列表
    async fn users(&self, ctx: &Context<'_>, query: UserQuery) -> Result<Vec<User>, Error> {
        let user_service = ctx.data::<UserService>()?;
        let users = user_service.get_users(&query).await?;
        Ok(users)
    }
}

#[derive(InputObject)]
pub struct CreateUserInput {
    pub username: String,
    pub email: String,
    pub password: String,
}

// Mutation定义
pub struct UserMutation;

#[Object]
impl UserMutation {
    // 创建用户
    async fn create_user(&self, ctx: &Context<'_>, input: CreateUserInput) -> Result<User, Error> {
        let user_service = ctx.data::<UserService>()?;
        let user = user_service.create_user(&input).await?;
        Ok(user)
    }
}
```

## API网关设计

### 1. 网关架构

```rust
// API网关
pub struct ApiGateway {
    routes: HashMap<String, Route>,
    middleware: Vec<Box<dyn Middleware>>,
    load_balancer: LoadBalancer,
}

impl ApiGateway {
    pub async fn handle_request(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        // 应用中间件
        let mut request = self.apply_middleware(request).await?;
        
        // 路由匹配
        let route = self.match_route(&request.path).await?;
        
        // 负载均衡
        let backend = self.load_balancer.select_backend(&route).await?;
        
        // 转发请求
        let response = self.forward_request(request, backend).await?;
        
        Ok(response)
    }
    
    async fn apply_middleware(&self, mut request: HttpRequest) -> Result<HttpRequest, Error> {
        for middleware in &self.middleware {
            request = middleware.process(request).await?;
        }
        Ok(request)
    }
}
```

### 2. 中间件实现

```rust
// 中间件trait
#[async_trait]
pub trait Middleware: Send + Sync {
    async fn process(&self, request: HttpRequest) -> Result<HttpRequest, Error>;
}

// 认证中间件
pub struct AuthMiddleware {
    auth_service: AuthService,
}

#[async_trait]
impl Middleware for AuthMiddleware {
    async fn process(&self, request: HttpRequest) -> Result<HttpRequest, Error> {
        let token = request.headers()
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "));
        
        if let Some(token) = token {
            let user = self.auth_service.validate_token(token).await?;
            request.extensions_mut().insert(user);
        }
        
        Ok(request)
    }
}
```

## API版本控制

### 1. 版本策略

```rust
// API版本管理器
pub struct ApiVersionManager {
    versions: HashMap<String, ApiVersion>,
    default_version: String,
}

impl ApiVersionManager {
    pub fn new() -> Self {
        let mut versions = HashMap::new();
        versions.insert("v1".to_string(), ApiVersion::new("v1"));
        versions.insert("v2".to_string(), ApiVersion::new("v2"));
        
        Self {
            versions,
            default_version: "v1".to_string(),
        }
    }
    
    pub async fn handle_request(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        // 从URL路径提取版本
        let version = self.extract_version(&request.uri().path())
            .unwrap_or_else(|| self.default_version.clone());
        
        if let Some(api_version) = self.versions.get(&version) {
            api_version.handle_request(request).await
        } else {
            Err(Error::UnsupportedVersion)
        }
    }
}
```

## API文档和测试

### 1. OpenAPI规范

```rust
// OpenAPI文档生成器
pub struct OpenApiGenerator {
    spec: OpenApiSpec,
    paths: HashMap<String, PathItem>,
}

impl OpenApiGenerator {
    pub fn new() -> Self {
        Self {
            spec: OpenApiSpec::new("User API", "1.0.0"),
            paths: HashMap::new(),
        }
    }
    
    pub fn add_path(&mut self, path: &str, method: &str, operation: Operation) {
        let path_item = self.paths.entry(path.to_string()).or_insert_with(PathItem::new);
        
        match method.to_uppercase().as_str() {
            "GET" => path_item.get = Some(operation),
            "POST" => path_item.post = Some(operation),
            "PUT" => path_item.put = Some(operation),
            "DELETE" => path_item.delete = Some(operation),
            _ => {}
        }
    }
}
```

### 2. API测试

```rust
// API测试框架
pub struct ApiTestFramework {
    client: TestClient,
    test_cases: Vec<TestCase>,
}

impl ApiTestFramework {
    pub async fn run_tests(&self) -> Result<TestResults, Error> {
        let mut results = TestResults::new();
        
        for test_case in &self.test_cases {
            let result = self.run_test_case(test_case).await?;
            results.add_result(result);
        }
        
        Ok(results)
    }
    
    async fn run_test_case(&self, test_case: &TestCase) -> Result<TestResult, Error> {
        let start_time = Instant::now();
        
        // 执行请求
        let response = self.client
            .request(test_case.method.clone(), test_case.url.clone())
            .headers(test_case.headers.clone())
            .body(test_case.body.clone())
            .send()
            .await?;
        
        let duration = start_time.elapsed();
        
        // 验证响应
        let assertions = self.validate_response(&response, &test_case.expected).await?;
        
        Ok(TestResult {
            name: test_case.name.clone(),
            success: assertions.all_passed(),
            duration,
            assertions,
        })
    }
}
```

## 最佳实践

### 1. 设计原则

- **一致性**: 保持API设计风格一致
- **简洁性**: 避免过度设计，保持简单
- **可预测性**: 遵循约定俗成的模式
- **向后兼容**: 保持API的稳定性

### 2. 性能优化

- **缓存策略**: 合理使用HTTP缓存
- **分页**: 大数据集使用分页
- **压缩**: 启用响应压缩
- **CDN**: 使用内容分发网络

### 3. 安全考虑

- **认证授权**: 实现适当的认证机制
- **输入验证**: 严格验证所有输入
- **HTTPS**: 强制使用HTTPS
- **速率限制**: 防止API滥用

### 4. 监控和分析

- **指标收集**: 收集API使用指标
- **错误监控**: 监控API错误率
- **性能监控**: 监控响应时间
- **使用分析**: 分析API使用模式

## 总结

良好的API设计是构建成功软件系统的关键要素。成功的API设计需要：

1. **用户导向**: 以API使用者为中心设计
2. **标准化**: 遵循行业标准和最佳实践
3. **文档化**: 提供清晰完整的文档
4. **测试驱动**: 通过测试确保质量
5. **持续改进**: 根据反馈不断优化

API设计将继续演进，与微服务、云原生、AI等新技术融合，为现代软件架构提供更强有力的支撑。
